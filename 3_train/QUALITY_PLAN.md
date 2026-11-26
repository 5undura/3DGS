# 화질 개선 계획 (2026-11-26, 재촬영 없이 먼저)

목표: 영상 스캔 → 3DGS → VR 에서 Hyperscape 급으로 선명하게. 대상은 복도 1, 실험은 한 번에 한 가지만.
기준선: Brush 1920 / 300만 / SH2 / 30,000 (그대로 보존). 결과는 모두 3_train/README.md 표에 기록.
옵션 이름은 리서치(문서·소스)에서 본 것이며 **실행 전 --help 로 다시 확인**.

## 왜 Hyperscape 보다 못한가 (요약)
- Hyperscape: 헤드셋으로 표면 가까이 5~10분 촬영(정확한 VIO 자세) → 서버 학습 → **클라우드 GPU 렌더링 스트리밍** (헤드셋 예산·압축 없음)
- 우리: 어두운 곳 영상(블러·롤링셔터·자동 노출) + 흰 벽(SIFT 특징점 부족 → 자세 오차) + 절반 해상도 학습 + Unity 원본 논문 렌더러
- 3DGS 는 자세 오차에 아주 민감: 회전 ±1° 노이즈로 SSIM 0.76 → 0.35 (arXiv 2403.20309). **흐림의 1순위 후보는 자세**

## 실측 (2026-11-26, 드라이브의 기존 결과로 확인)
- Colab gsplat MCMC 960px 평가(학습 경로 위 8장마다 1장): **방 6 PSNR 30.1 / 복도 1 24.5** (SSIM 0.915 / 0.866, LPIPS 0.16 / 0.25).
  복도는 VR 이 아니라 **학습 시점 근처부터 재현이 안 됨** → 원인은 정합·블러·노출 쪽. Difix 후 25.2 (+0.7) 뿐
- 복도 1 궤적(1084장 중 136장 표본): 벽 정면(60도 이상) 48%, 천장 20%, 바닥 26%, 높이 범위 약 1.5m, 재방문 많음
  → **촬영 커버리지는 문제 아님**
- 같은 표본에서 촬영 순서상 **중앙 속도의 6~21배로 순간이동 + 30~155도 회전** 7곳 (예: f_00779→f_00795, f_00815→f_00831, f_01871→f_01903).
  복도 양 끝 창문처럼 비슷한 장면이 많아 **일부 사진이 엉뚱한 위치에 정합됐을 가능성** (예전 exhaustive 매칭). 전체 1084장으로 다시 확인 필요
- 실제 4K 프레임 4장: 색은 정상 SDR(HLG 아님, 노란 조명), 노이즈 낮음(강한 노이즈 제거), **실효 해상도 가로 약 2,500~2,800px**
  → 학습 1920 은 디테일 일부 손실, 3840 은 이득 적음 → **2560 이 적정**
- Osmo Pocket 3: 전자식 흔들림 보정 없음(기계식 짐벌). 대신 연속 AF·**브리딩 보정**·줌이 화각을 바꿈.
  어두우면 셔터가 1/30 까지 느려져 블러, 고 ISO 노이즈 제거가 디테일을 뭉갬 (Pro 모드 수동 노출·Sharpness/NR −2 가능)
- Difix: 576x1024 모델, 8GB/12GB 에서 사실상 어려움, PLY 에 안 구워짐 → 우선순위 낮춤
- LichtFeld v0.5.4.3 릴리스 노트에 `--pose-opt`(자세 최적화) 추가 → 설치 버전 `--help` 로 확인 후 1순위 실험

## 0. 진단 (GPU 거의 안 씀, 반나절)
```
python tools\diagnose.py --video <영상>\1.MP4 --ffprobe F:\Sample\tools\ffmpeg_tool\ffprobe.exe
python tools\diagnose.py --frames F:\Sample\4차\1_프레임\1a --model F:\Sample\4차\2_정합데이터\1b\data
```
- HLG/PQ 경고 → SDR 변환부터 (색 물빠짐·저대비). 대비 < 80 이면 로그/HDR 의심
- "960px 로는 상위 절반인데 원본에서는 하위 25%" 비율이 크면 → 추출에 `--sharp-res 1920`, `--abs-floor-pct 10`
- 재투영 오차 > 1px, 사진당 3D 점이 적은 사진 다수 → 정합 개선(2단계)
- 벽 정면 사진 < 15%, 천장 < 5%, 한 높이만 → 재촬영 없이는 한계 (Difix·깊이 prior 로 일부 보완)
- 촬영 설정 확인: Osmo 의 전자식 흔들림 보정(RockSteady/HorizonSteady)·디워프가 켜져 있었으면 프레임마다 화각이 바뀌어 단일 카메라 가정이 깨짐

## 1. 자세가 원인인지 확인 (1구간, 1~2시간)
- LichtFeld 를 v0.5.4.3 이상으로 올렸고 `--help` 에 `--pose-opt` 가 있으면: 같은 설정으로 켬/끔 두 번 (`$TrainExtraArgs`)
- 없으면 gsplat `examples/simple_trainer.py mcmc` 로 `--pose_opt` 켬/끔 (설치는 PIPELINE_UPGRADE.md ③′)
- 켠 쪽 PSNR 이 크게 오르면(예: +2dB 이상) 자세가 원인 → 2단계 정합 개선이 가장 큰 효과
- 순간이동 사진(diagnose.py 가 출력) 앞뒤를 `*_skip` 으로 빼고 학습한 것과도 비교

## 2. 정합 개선 (자세가 원인이면 가장 큰 효과)
config.ps1 (새 작업 폴더에서):
```
$FeatureArgs = @("--FeatureExtraction.type", "ALIKED_N16ROT")
$MatcherArgs = @("--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--SequentialMatching.overlap", "20", "--SequentialMatching.loop_detection", "1")
```
- `feature_extractor -h` 에 FeatureExtraction.type 이 없으면(ONNX 없는 빌드) SIFT 유지 +
  `--SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1`, 매칭 `--FeatureMatching.guided_matching 1`
- `$Mapper = "incremental"` 과 `"global"` 을 한 구간에서 비교 (재투영 오차·등록률·천장 점 수·순간이동: diagnose.py --model)
- sequential 매칭의 loop detection 도 비슷한 장면을 잘못 이을 수 있음 → 순간이동이 나오면 loop_detection 끈 것과 비교
- 대안: **RealityScan 2.0** (학생 무료, 특징 적은 면 정합 개선, Export > Registration 으로 COLMAP 텍스트 내보내기, 왜곡 편 사진도 같이 내보내야 함).
  Epic Games Launcher 설치가 필요하므로 설치 전에 확인

## 3. 학습 (한 가지씩)
1. LichtFeld `--ppisp --bg-mode=modulation` (train_all.ps1 기본값. 자동 노출 흔들림 + 어두운 곳 투명화 대응)
2. 해상도: `--max-width=2560` (실효 해상도가 약 2,500~2,800px 이라 그 이상은 이득 적음). (`$TrainExtraArgs`, 3060 Ti 는 메모리 부족하면 `--max-cap` 을 낮춤)
3. 개수 400~600만 (`--max-cap`), 전략 `--strategy igs+` 와 `mcmc` 비교 (igs+ 는 세부가 선명하나 촬영이 나쁘면 흰 안개)
4. 그래도 흰 벽·천장이 안개 같으면: DN-Splatter(nerfstudio, WSL2) 단안 깊이·법선 prior
5. 블러·롤링셔터가 크면: SpectacularAI 3dgs-deblur 또는 BAD-Gaussians (nerfstudio, WSL2, 학습 약 5배 느림)
6. Difix 는 8GB/12GB 에서 사실상 어려워 후순위
- `--enable-mip` / gsplat `antialiased` 는 끔 (aras-p 렌더러가 지원 안 함)
- "반투명 95%" 는 MCMC·Brush 의 opacity 정규화 때문에 원래 높게 나옴 → 숫자보다 **빨간/분홍 배경에서 비치는지**로 판단

## 4. VR 쪽 (AVT, 영향 확인만)
- aras-p 품질 프리셋 Very High/High 인지 (Low 이하는 PSNR 35 로 뭉갬), 구간 하나만 넣어 A/B
- 배경 메쉬(avt_backdrop) 끈 화면과 비교: 스플랫이 반투명하면 실제로는 메쉬 텍스처가 보이는 것
- aras-p v2.0.0 은 VR 경로가 깨졌을 수 있다고 작성자가 경고 → 버전 고정. 대안 A/B: ninjamode/Unity-VR-Gaussian-Splatting, wuyize25/gsplat-unity

## 재촬영 시 (위로도 부족하면)
Pro 모드: 노출 M(셔터 1/250+), ISO·WB 고정, 초점 고정(연속 AF 끔), 브리딩 보정·줌 끔, Sharpness/NR −2, 색 Normal, 조명 보강, 높이 2~3단 왕복, 벽 정면·천장·바닥 패스, 천천히 + 루프, 흰 벽에 임시 마커
