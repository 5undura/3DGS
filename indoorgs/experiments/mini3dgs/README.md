# mini3dgs — CPU 전용 작은 3DGS (평가 도구)

GPU 가 없는 곳에서 **벽 평탄화(walls/refine·seed·snap) 같은 변경이 3DGS 품질을 올리는지 상대 비교**하려고 만든
순수 torch(autograd) 3DGS. 실제 결과물용 학습기(Brush / LichtFeld)를 대신하지 않는다.

| 파일 | 내용 |
|---|---|
| `gs.py` | 래스터라이저(투영·타일·합성), SH, SSIM/PSNR, PLY 입출력(`indoorgs.plyio`, Brush 순서), `render()` |
| `train.py` | 장면 읽기(COLMAP 모델 + 사진), 학습, 평가, `vr_cameras()`, CLI |
| `toyroom.py` | 4 x 3 x 2.6 m 상자 방 장면(광선 추적 정답, COLMAP 모델 파일) + 확인 실행 |

## 사용

```bash
PY=.../scratchpad/venv/bin/python        # torch CPU
$PY train.py --model <.../sparse/1> --images <사진 폴더> --out <결과> --iters 1500 --res 200 [--init <점 모델 폴더|PLY>]
$PY train.py ... --init-ply <학습된/고친 PLY> --iters 1000 --checkpoints 300,600   # 이어 학습 (fine-tune: 밀도 조절 없음, means lr 끝 값)
$PY toyroom.py --out <폴더> --iters 1500 --res 200          # 확인 실행
```

```python
from train import train, vr_cameras, load_scene
from gs import render, Camera
m = train(model_dir, image_dir, out_dir, iters=1500, res=200,
          init_points=None,          # None = 모델의 points3D / (xyz, rgb) / 모델 폴더 / 점 PLY (seed.write_seeded_model 결과 등)
          novel_views=cb,            # cb(train_cams, test_cams) -> [{"name", "camera": Camera, "image": RGB}]
          bg=(0,0,0), random_bg=False, cap=100_000, sh_degree=2)
img = render("out/point_cloud.ply", cam)     # (H,W,3) float, with_depth=True 면 alpha·depth 도
```

- `--model` 은 **모델 폴더 자체**(예: `sparse/1`)를 주는 것이 안전하다. `sparse` 를 주면 `find_model` 이 `sparse/0`
  을 고르는데, 합성 복도 실행들에서는 `sparse/0` 이 2~3장짜리 잘못된 조각이다.
- 사진은 원본 크기 그대로 두면 된다. 학습기가 가로 `res` 로 줄이고(INTER_AREA) K 도 같이 줄인다.
  SIMPLE_RADIAL/RADIAL/OPENCV 는 `cv2.undistort` 로 같은 K 에 편다 (PINHOLE/SIMPLE_PINHOLE 은 그대로).
- 테스트 = 이름순 `i % 8 == 0` (원본 3DGS 와 같음).
- 새 시점: 장면 제공자가 카메라와 정답을 준다. 카메라는 **COLMAP 모델 좌표**여야 하므로, 합성 복도처럼 정답 좌표가
  따로 있는 장면은 제공자가 Sim3(모델->정답)으로 카메라를 옮겨 정답을 렌더해야 한다.
  `vr_cameras(cams, offset=0.3, grazing_deg=60, up=None)` 가 카메라마다 6개(옆 ±0.3, 위·아래 ±0.3, 위 축 기준
  ±60도 회전 = 벽을 비스듬히 봄)를 만든다. offset 은 모델 단위(축척이 m 가 아니면 환산해서 줄 것).
- 결과 `out/`: `point_cloud.ply` (Brush 속성 순서, `walls.snap` 이 그대로 읽음), `metrics.json`
  (시점별 PSNR/SSIM/see_through, 평균, `ply_stats` = 개수·반투명 비율·바늘 비율, 시간, 밀도 조절 기록),
  `renders/{test,novel}_*.png` (정답 | 렌더 | 누적 alpha).
- `see_through` = 누적 alpha < 0.95 인 픽셀 비율 (배경이 비쳐 보이는 정도). 검정 배경으로 평가.

## 구현 (원본 3DGS 와 같은 것)

- Σ = R S Sᵀ Rᵀ (쿼터니언 + log 스케일), EWA 투영 J W Σ Wᵀ Jᵀ, 화면 밖 1.3·tan(fov/2) 자르기, **+0.3 px 팽창**,
  반경 3σ, 근평면 0.2, α = min(0.99, o·exp(-½dᵀΣ'⁻¹d)), α < 1/255 버림, 깊이 정렬 앞->뒤 합성 + 배경색.
- SH 차수 2 까지 (원본과 같은 기저·부호). 학습 중 0 -> 2 로 올림.
- 16x16 타일, 3σ 경계 상자로 가우시안을 타일에 배정, 타일을 개수 순으로 묶어 패딩 [B, 256, K] 텐서로 합성.
  메모리가 크면 묶음마다 `torch.utils.checkpoint`.
- 손실 0.8·L1 + 0.2·(1-SSIM 11x11), Adam(eps 1e-15), 원본 학습률 (means 1.6e-4·extent 지수 감소 -> 1%).
- 초기화: 점마다 가장 가까운 3 이웃 거리로 등방 스케일, 불투명도 0.1.
- 밀도 조절: 화면 위치 기울기(NDC 단위, 원본과 같은 환산) 평균 > 0.0002 이면 복제(작은 것) / 2개로 분할(큰 것, 1/1.6),
  불투명도 < 0.005 제거, 불투명도 리셋 후에는 화면 반경 > 20px·크기 > 0.1·extent 제거. 개수 상한 `cap` 을 넘으면
  기울기 큰 후보부터 남은 자리만큼만.
- 검증: 타일 결과 = 타일 없는 직접 합성 (최대 차 2e-7), autograd 기울기 = 중앙 차분 (float64, 상대 오차 < 1e-7).

## 한계 (솔직하게)

- **해상도·규모가 작다**: 200~320 px 폭, 수만 개 가우시안, 수천 반복. 실제(1920 px, 300만 개, 30k 반복)와 절대값이
  다르다. 쓸 수 있는 것은 **같은 조건의 A/B 비교**(초기 점, snap 전후 등)뿐. 작은 해상도에서는 흰 벽의 미세한
  울퉁불퉁함·얇은 반투명 층이 덜 보일 수 있다.
- 일정은 30k 기준을 반복 수에 비례해 줄인 것. 4000 반복 미만이면 **불투명도 리셋을 끔** (짧은 학습에서 리셋하면
  회복할 시간이 없음). 그래서 반투명 비율(`translucent_ratio`)이 실제 학습기보다 높게 나온다 (상자 방 0.55).
  비교할 때는 같은 설정끼리만.
- Brush/LichtFeld 고유 기능 없음: MCMC 재배치, 안티에일리어싱(mip), 노출 보정, 비등방 정규화, 조기 종료 등.
  밀도 조절은 원본 3DGS(2023) 방식이고 Brush 기본값과 다르다.
- 타일 안 조기 종료(T < 1e-4)를 하지 않음 (결과 차이는 무시할 수준, 대신 느림). 정렬은 가우시안 중심 깊이 기준
  (원본과 같음 — 겹치는 가우시안 사이의 깊이 순서 튐은 원본과 같은 수준으로 있음).
- 왜곡 카메라는 미리 펴서 씀 (원본 3DGS 처럼 image_undistorter 를 거친 것과 비슷하지만 같은 K 를 유지해 가장자리
  일부가 검게 될 수 있음).
- CPU 속도: **CPU 가 바쁠 때 torch 다중 스레드는 매우 느려진다** (부하 11 / 4코어에서 4 스레드 4.5 s/it, 1 스레드
  0.27 s/it). 기본값은 빈 코어 수만큼 자동 (`threads=None`), 필요하면 `--threads`.

## 확인 실행 (상자 방, 2026-11-27)

`toyroom.py --iters 1500 --res 200` (CPU 1 스레드, 다른 작업으로 부하 ~11 인 상태)

- 장면: 40 장 320x240 (학습 35 / 테스트 5), COLMAP 비슷한 초기 점 5,985 개 (무늬 쪽에 몰림, 1.5 cm 잡음, 2% 허공 점),
  새 시점 30 장 (테스트 5 장 x 6, 방 밖 제외).
- 테스트 PSNR: 반복 250 -> 24.0, 750 -> 29.6, 1000 -> 31.8, 1250 -> 32.4, **1500 -> 32.6 dB** (SSIM 0.976, 비침 0.003)
- 새 시점(VR) PSNR **32.2 dB** (SSIM 0.967). 위·아래로 옮긴 시점에서 포스터 뒤 줄무늬 같은 실제 3DGS 와 비슷한 결함이 보임.
- 가우시안 5,985 -> 36,202 개, 반투명 비율 0.546, 바늘 비율 0.012.
- 시간: 학습 575 s (0.38 s/it 평균; 6k 개일 때 0.18, 36k 개일 때 0.48 s/it), 전체 590 s.

## 합성 복도 연결 확인 (읽기만, 2026-11-27)

`train.py --model synth/v1/runs/r2_seq/sparse/1 --images synth/v1/images --iters 400 --res 256`
(SIMPLE_RADIAL, 238 장 -> 학습 208 / 테스트 30, 초기 점 7,516, 256x144): 읽기 1.4 s, **0.54 s/it** (7.5k 개, 1 스레드),
400 반복(밀도 조절 거의 없음) 테스트 PSNR 21.6. 3000 반복 + 밀도 조절이면 가우시안이 늘어 대략 1 s/it, 약 50 분 예상.
