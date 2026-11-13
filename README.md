# NoClip 3DGS 파이프라인

촬영 영상 → 프레임 추출 → COLMAP 정합 → Brush 학습 → PLY 까지의 작업 스크립트.
영상·프레임·PLY 같은 데이터는 저장소에 올리지 않고 클라우드(구글 드라이브 `3DGS` 폴더)에 보관함.

## 준비물
- **COLMAP (CUDA 빌드)** — Windows용 `colmap-x64-windows-cuda`. CUDA 없는 빌드는 정합이 몇 배 느림
- **Brush** — `brush_app.exe`
- **Python 3** + `pip install opencv-python numpy`

## 처음 설정
```powershell
git clone https://github.com/5undura/3DGS.git
cd 3DGS
copy config.example.ps1 config.ps1   # 그다음 config.ps1 을 열어 자기 PC 경로로 수정
```
PowerShell에서 스크립트 실행이 막히면 한 번만: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 폴더 규칙
작업 폴더(`$Work`)에 영상을 `1.MP4`, `2.MP4` … 처럼 구간 번호로 넣어 둠. 구간마다 결과가 이렇게 생김:

| 폴더 | 내용 |
|---|---|
| `Na` | 추출한 프레임 (`f_<순번>_<원본프레임번호>.jpg`) |
| `Na_skip` | 솎아내서 정합에 안 쓴 프레임 (삭제 안 하고 보관) |
| `Nb\colmap` | COLMAP 작업 파일 (`db.db`, `sparse\0`, `sparse\1` …) |
| `Nb\data` | undistort 결과 — **Brush에 넣는 폴더** |
| `Nc` | Brush 결과 PLY |
| `night_log.txt` | 전체 진행 기록 (구간별 프레임 수, 정합 시간, 모델별 등록 장수) |

구간은 영상을 나눠 찍은 단위. 똑같이 생긴 방끼리 섞여서 정합이 꼬이는 걸 막으려고 나눠서 처리함.

## 실행 순서
```powershell
.\1_extract\extract_all.ps1     # 1. 프레임 추출 (구간당 수 분~수십 분)
.\2_register\register_all.ps1   # 2. 정합 (구간당 수 시간, 밤새 돌리기)
```
3. Brush 학습 → [3_train/README.md](3_train/README.md)

두 스크립트 모두 이미 끝난 구간은 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 진행됨.

### 1. 프레임 추출 (`1_extract/pick_best.py`)
초당 몇 장씩 고정으로 뽑지 않고, **선명하면서 직전에 뽑은 사진과 충분히 다른 프레임**만 저장함.
- 흔들린 프레임 제외 (주변 프레임 선명도 중간값의 75% 미만이면 버림)
- 1초 이상 아무것도 안 뽑히면 그 구간에서 가장 선명한 프레임을 강제로 저장 (구멍 방지)
- 장수 조절: `--min-diff` 를 키우면 적게, 줄이면 많이 뽑음 (기본 6)

### 2. 정합 (`2_register/register_all.ps1`)
1. 프레임이 1200장을 넘으면 약 1100장이 되도록 일정 간격으로 솎아냄
2. 특징점 추출 (카메라 1개, OPENCV 렌즈 모델) → 매칭 (기본 exhaustive) → mapper
3. 정합 결과가 여러 모델로 쪼개지는 경우가 많음 → **등록된 사진이 가장 많은 모델**을 자동 선택
4. 그 모델로 image_undistorter 실행 → `Nb\data`

## 참고 기록 (2차 촬영, Osmo Pocket 3)
| 구간 | 추출 | 정합 사용 | 등록 |
|---|---|---|---|
| 복도 1 | 2,199 | 1,100 | 1,084 |
| 복도 2 | 1,888 | 944 | 919 |
| 방 3 | 2,004 | 1,002 | 1,002 |
| 방 4 | 847 | 847 | 826 |
| 방 5 | 1,130 | 1,130 | 1,125 |
| 방 6 | 470 | 470 | 470 |

RTX 3060 Ti 기준 exhaustive 정합 전체 약 45시간.

## 주의사항
- **`sparse\0` 이 정답이 아닐 수 있음.** 사진 몇 장짜리 조각 모델이 0번이 되는 경우가 많음. 스크립트가 가장 큰 모델을 고르지만, 직접 할 때도 꼭 확인
- **4K로 학습하면 무너질 수 있음.** 1920으로 낮춰서 학습
- **촬영:** 벽을 정면으로 보는 구간 넣기, 노출·화이트밸런스 고정, 셔터 1/100초 이상. 앞만 보고 찍으면 벽에 줄무늬가 생김
- **마커:** 단색 벽에는 번호를 서로 다르게 적은 빨간 A4 종이를 2~3m 간격으로 붙이기

## 구글 드라이브 `3DGS` 폴더 구조
영상·정합 데이터·PLY 는 저장소 대신 팀 공유 드라이브 폴더에 둠.

| 폴더 | 내용 |
|---|---|
| `1_촬영영상/3차` | 3차 촬영 (`3.zip`) |
| `1_촬영영상/4차` | 4차 촬영 원본 영상 (`DJI_*.MP4`, 7개) |
| `2_정합데이터/4차/1~6` | 구간별 `images` + `sparse` (= PC 의 `Nb\data`). Brush · Colab 학습 입력 |
| `2_정합데이터/4차/6_colab사본` | 방 6 Colab 실험용으로 따로 올린 사본 (`6` 과 같은 데이터) |
| `3_학습결과/3차_Brush` | 3차 촬영 Brush 결과 PLY (구간 1~7) |
| `3_학습결과/4차_Brush_기본` | 4차 Brush 기본 설정 PLY (1~6, `1_1`·`2_2` 는 600만 개 버전) |
| `3_학습결과/4차_Brush_E설정` | Brush 크기 벌점 + 성장 강화 실험 (해결 방향 3) |
| `3_학습결과/4차_colab/out/<구간>` | Colab 실험 결과: `ply/`(Difix base·difix), `gs2d/ply/`(2DGS), `compare.png` |
| `4_게임빌드` | Unity VR 빌드 |

## colab (Difix · 2DGS · 배경 무작위화 실험)
구글 Colab 에서 노트북을 열고 **런타임 유형을 T4 GPU** 로 바꾼 뒤 **런타임 → 모두 실행**. 결과는 드라이브 `3DGS/3_학습결과/4차_colab/out/<구간 이름>` 에 저장됨.

| 노트북 | 내용 |
|---|---|
| `colab/gs2d_colab.ipynb` | 2DGS(공식 구현 `hbb1/2d-gaussian-splatting`) 학습 → 3DGS 형식 PLY 변환. 데이터 위치는 1번 셀의 `DRIVE_DATA` 하나만 지정 |
| `colab/difix_colab.ipynb` | gsplat(MCMC, 300만 상한) 기본 학습 → NVIDIA Difix 보정 학습 (+ 배경 무작위화 선택). 데이터 위치는 1번 셀의 `DRIVE_DATA`. T4 기준 약 4~5시간, 끊기면 이어서 함 |

데이터는 Brush 에 넣던 `Nb\data` 폴더(`images`, `sparse`)를 드라이브에 그대로 올리면 됨.

**gs2d_colab 이 알아서 처리하는 것**
- GPU 런타임이 아니거나 데이터 폴더가 틀리면 첫 셀에서 멈추고 이유를 출력
- CUDA 컴파일(`diff-surfel-rasterization`, `simple-knn`, 5~10분). 이미 설치돼 있으면 건너뜀. 최신 컴파일러용 `#include` 보강, simple-knn 은 gitlab 이 막히면 GitHub 미러 사용. 실패하면 오류 원인을 출력
- 로컬 복사본만 960 폭으로 축소 (4K 그대로 읽으면 Colab RAM 12GB 초과로 `^C` 와 함께 끊김)
- 5,000 스텝마다 체크포인트 → 런타임이 끊겨도 다시 "모두 실행" 하면 이어서 학습. 끝나면 체크포인트 자동 삭제
- 2DGS PLY(크기 값 2개)를 SuperSplat/Unity 용 3DGS 형식으로 변환 (`colab/nc_ply2d_to_3d.py` 와 같은 내용, 이 스크립트는 PC 에서도 numpy 만으로 실행됨)

**주의**
- 런타임이 새로 켜질 때마다 설치(컴파일)를 다시 해야 함
- 무료 Colab 은 GPU 사용 한도가 있음. 끝나면 **런타임 → 런타임 연결 해제 및 삭제**. 한도 소진 시 반나절~하루 뒤 풀림
- Unity 플러그인(aras-p)은 3DGS 만 지원하므로 2DGS 결과는 "줄무늬가 줄어드는지" 확인용
- gsplat(`difix_colab`)도 처음 학습할 때 10~30분 컴파일함 ("setting up cuda" 메시지)
- `difix_colab` 도 데이터 확인, 설치 건너뛰기, 5,000 스텝마다 저장·이어하기가 들어 있음. 4K 원본은 로컬에 복사하지 않고 줄인 사진(`images_4`)만 만듦
- 방 6 실험에서 gsplat 기본(default) 방식은 floater 가 많아 Brush 보다 나빴음 → MCMC 로 변경
- 2DGS, Difix 는 모두 비상업 연구용 라이선스

## tools
| 파일 | 내용 |
|---|---|
| `tools/flatten.py` | PLY 후처리: 큰 평면(벽·바닥·천장) 근처 스플랫을 납작하게 (효과 미미했음) |
| `tools/flatten2.py` | 위와 같은데 구역별 국소 평면 버전 (휜 복도용) |
| `colab/nc_ply2d_to_3d.py` | 2DGS PLY → 3DGS 형식 PLY 변환 |
| `tools/ply_stats.py` | PLY 여러 개의 스플랫 통계 비교 (개수, 바늘형 비율, 크기, 반투명 비율) |
| `3_train/brush_experiment.ps1` | 여러 구간을 같은 실험 설정으로 연속 학습 |
