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

## colab (Difix · 배경 무작위화 실험)
`colab/difix_colab.ipynb` 를 Google Colab 에서 열어 위에서부터 실행. 데이터는 구글 드라이브 `3DGS/colab/<방 이름>` 에 `images`, `sparse` 를 복사해 두면 됨. `nc_*.py` 는 노트북 안에도 들어 있음.

## tools
| 파일 | 내용 |
|---|---|
| `tools/flatten.py` | PLY 후처리: 큰 평면(벽·바닥·천장) 근처 스플랫을 납작하게 (효과 미미했음) |
| `tools/flatten2.py` | 위와 같은데 구역별 국소 평면 버전 (휜 복도용) |
| `tools/ply_stats.py` | PLY 여러 개의 스플랫 통계 비교 (개수, 바늘형 비율, 크기, 반투명 비율) |
| `3_train/brush_experiment.ps1` | 여러 구간을 같은 실험 설정으로 연속 학습 |
