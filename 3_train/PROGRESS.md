# 진행 일지 (QUALITY_PLAN 실행, 2026-11-27 밤~)

목표: 지금 영상 데이터로 VR 에서 깨끗하고 빈틈없는 스플랫. 대상 복도 1. 실험 결과 표는 `3_train/README.md`.

## 지금 상태
- [x] 저장소 `pipeline-upgrade` 클론 → `F:\Sample\tools\3DGS_repo` (예전 `F:\Sample\tools\pipeline` 은 git 아님, 그대로 둠)
- [x] 0단계 도구 버전·옵션 확인
- [x] 0단계 diagnose.py (원래 정합 1b, 복도 1 전체) → 아래 "진단 결과"
- [ ] 1단계 자세 확인 — **LichtFeld 없음 → 막힘** (아래 시행착오)
- [x] 2단계 재정합 (sequential + loop, global / incremental) → global 못 씀, **incremental 이 원래보다 나음** (아래 표)
- [x] 3단계 Brush 2560 → 1920 과 PSNR 같음 (해상도는 병목 아님)
- [x] 빈 곳 채우기 (tools/fill, `up\1b_origskip` 기준) → `up\1c_fill_origskip\fill_with_backdrop.ply`. PSNR 25.75→25.51, 반투명 86.9→85.0% (바탕층 포함 79.6%). **VR/SuperSplat 에서 천장·빈 곳 눈으로 확인 필요**
- [ ] LichtFeld pose-opt / PPISP — 빌드 성공(14:07), L1(pose-opt) +0.28, **L2(+mrnf) 26.51 / 반투명 63% = 지금까지 최고**. L3(노출 보정 방식) 진행 중

## 도구 확인 (2026-11-27)
| 도구 | 버전 | train_all / register_all 옵션 대조 |
|---|---|---|
| COLMAP | 4.2.1 (bd1fcf6, CUDA) | ImageReader.single_camera·camera_model, SequentialMatching.overlap·loop_detection, view_graph_calibrator, global_mapper 모두 있음. FeatureExtraction.type(ALIKED) 옵션도 있음(GPU용 DLL 없을 수 있음) |
| Brush | 0.3.0 | --total-steps, --max-splats, --sh-degree, --max-resolution, --export-every, --export-path 맞음. --eval-split-every / --eval-every 로 PSNR 평가 가능. **자세 최적화 옵션 없음** |
| LichtFeld | **없음** | `F:\Sample\tools\lfs` 에 build_log.txt 만. 11-26 빌드 시도에서 Strawberry Perl, CUDA 12.8, VS2022 C++ 없음으로 중단 (VS 는 2026 Community 만 있음) |
| Python | 3.12.7, OpenCV 5.0.0, numpy 2.5.4 | |
- COLMAP 은 한글 경로(F:\Sample\4차\up)에서 정상 동작 확인 (3장 feature_extractor 시험)
- GitHub 인증 정보가 이 PC 에 저장돼 있지 않아 push 는 자고 있는 동안 못 함 → 로컬 커밋만 쌓고 나중에 push

## 진단 결과 (복도 1, 원래 정합 4차 1b, 전체 1084장) — `F:\Sample\4차\up\diag\diag_1_orig.txt`
- 영상: HEVC 10bit 4K 23.976fps 64Mbps, **bt709 SDR (HLG/로그 아님)**, 대비 150 정상. 표본 간 밝기 53~187 → 자동 노출 흔들림 있음 (노출 보정 필요, Brush 엔 없음)
- 실효 해상도 가로 약 **2,688px** → 학습 2560 이 적정 (계획과 같음)
- 프레임(1a 1100장): 960px 에서만 선명한 프레임 0% → 추출(pick_best)은 문제 아님
- 정합: PINHOLE(왜곡 편 뒤), 1084/1100 장 등록, 점 64.5만, **재투영 오차 1.05px (>1px 경고)**, 사진당 점 100개 미만 18장
- 촬영: 높이 표준편차가 길이의 0.8% (거의 한 높이), 벽 정면 52%·천장 19%·바닥 27% → 커버리지는 괜찮음
- **순간이동: 중앙 속도 5배 넘게 튄 곳 17곳** (클라우드 표본의 7곳보다 많음). 최대 229배(f_01729→f_01731), 회전 169도(f_00005→f_00007).
  몰려 있는 구간: f_00679~00707, f_00741~00817, f_01729~01733, f_01885~01889
  → 상위 15곳에 걸린 사진 25장을 뺀 모델 `up\1b_origskip\data` 생성 (COLMAP image_deleter, 1059장, 사진은 원본 연결)

## 재정합 결과 (같은 프레임 1100장, sequential overlap 20 + loop detection, 매칭 25분)
| 정합 | 등록 | 3D 점 | 재투영 오차 평균/중앙 | 점 100개 미만 사진 | 순간이동(5배↑) | 시간 | 판정 |
|---|---|---|---|---|---|---|---|
| 원래 4차 1b (exhaustive?, incremental) | 1084 | 64.5만 | 1.05 / 1.02 | 18 | 17 | - | 기준 |
| global (GLOMAP) `up\1b_global_skip` | 1082 | 48.8만 | 1.04 / 0.92 | 65 | **100** | 46분 | **못 씀**: f_00437~00453 9장이 수천 단위 밖으로 튐(궤적 길이 15→3197), 복도 중간 f_00745~00825 도 크게 튐 |
| incremental `up\inc\1b` | 1072 | 54.3만 | **0.96 / 0.92** | 31 | **12** | 281분 | **원래보다 나음** (재투영 1px 미만, 순간이동 17→12). 남은 12곳 중 상당수는 22~45 프레임 간격(사이 프레임 미등록)이라 일부는 실제 이동일 수 있음. Brush 평가 진행 중 `up\inc\1c_brush` |

## Brush 비교 (기준선 설정, 평가 8장마다 1장)
- 원래 정합: PSNR 25.50 / SSIM 0.872. **PSNR 최하위 사진이 순간이동 목록과 겹침** (f_00743 9.2, f_01887 13.2, f_00779 14.5, f_00815 16.3)
- 순간이동 25장 제외: 25.75 / 0.874 (+0.25). 평가 사진 구성이 바뀌어(공통 1장) 정확한 비교는 아님.
  최하위 f_00751 9.3 → 뺀 사진 주변에 아직 잘못 놓인 사진이 남아 있음 (진단은 상위 15곳만 출력)
- 2560 (순간이동 제외 모델): 2560 평가 25.66 / 0.873, **1920 으로 줄여 평가하면 25.76 / 0.874 → 1920 과 같음**. 바늘형(>10배) 16.3→13.2%, 반투명 86.9→85.4%, 시간 36→54분.
  → 해상도는 병목 아님. VR 에서 가까이 볼 때 차이가 있는지는 눈으로 확인 필요 (SuperSplat 에서 같은 위치 비교)
- incremental 재정합: 25.07 / 0.862. **공통 평가 사진 32장 기준 원래 25.76 → 25.07 (나빠짐)**, 반투명 89.0%.
  COLMAP 수치(재투영·순간이동)가 좋아져도 학습 결과는 안 좋아짐 → 3D 점이 적고(54만 vs 64.5만) 매칭이 sequential 이라 사진 간 연결이 약한 탓으로 추정.
  → **원래 정합 + 순간이동 25장 제외(`up\1b_origskip`)를 기준 정합으로 확정**
- 빈 곳 채우기 (순간이동 제외판 + tools/fill): 같은 평가 133장 25.75 → 25.51 / SSIM 0.874 → 0.871.
  반투명 86.9 → 85.0%, 바늘형(>10배) 16.3 → 13.6%, 바탕층 합친 PLY 는 반투명 79.6%.
  PSNR 은 학습 사진 시점만 재므로 "빈 곳"(사진에 안 찍힌 천장 등) 개선은 안 잡힘 → 약간의 하락은 보강 점이 학습을 조금 방해한 정도로 보임. 판단은 눈으로.
  `fill_section.py` 가 한글 경로 사진을 못 읽던 버그(cv2.imread) 고침.
- 해석: 순간이동 사진은 실제로 잘못 놓였지만 전체 PSNR 에 주는 영향은 작음 → 복도 PSNR 25 의 주원인은 다른 곳(전반적 자세 정밀도 1px, 노출 흔들림, 흰 벽) 일 가능성. LichtFeld pose-opt + PPISP 로 확인 예정

## 시행착오
- LichtFeld `--pose-opt` 비교(1단계)는 설치(VS2022 C++ 워크로드, CUDA 12.8, Strawberry Perl)가 필요해 보류. 규칙상 설치는 먼저 물어봄.
  대안 gsplat `--pose_opt` 도 torch+gsplat 설치가 필요해 보류.
  - 2026-11-27: 사용자가 Strawberry Perl, CUDA 12.8, VS2022 C++ 설치 중 → 끝나면 확인 후 LICHTFELD_SETUP.md 대로 v0.5.4.3 빌드, pose-opt 비교 진행
  - 03:50 설치 확인됨 (VS2022 Community + 영어 팩, CUDA 12.8, Strawberry Perl, CMake 3.31.6/Ninja 는 VS 포함).
    `F:\Sample\tools\lfs\{vcpkg, LichtFeld-Studio(v0.5.4.3)}` 클론, 소스 argument_parser.cpp 에 `--pose-opt` 있음 확인.
    빌드 스크립트 `F:\Sample\tools\lfs\build_lfs.bat` (CUDA arch 86 = 3060 Ti 전용. 5070 PC 에서 쓰려면 "86;120" 으로 다시 빌드), 로그 `build_log2.txt`. vcpkg 가 의존성을 받아 빌드하느라 오래 걸림
  - 06:00 **빌드 실패**: vcpkg 의존성 libplacebo(LichtFeld 오버레이 포트)가 Windows 에서 clang-cl 을 요구
    ("Install the 'C++ Clang Compiler for Windows' individual component"). 이 PC 에 clang-cl 없음 → **VS2022 Installer > 개별 구성 요소 > "Windows용 C++ Clang 컴파일러" 설치 필요 (사용자 확인 대기)**.
    다른 의존성(ffmpeg, python3 등)은 빌드됨. C: 의 vcpkg 바이너리 캐시 저장이 "다른 드라이브로 옮길 수 없음"으로 실패 → build_lfs.bat 에 VCPKG_DEFAULT_BINARY_CACHE=F:\Sample\tools\lfs\vcpkg_cache 추가
  → 그동안 "자세가 원인인지"를 **재정합 전후 / 순간이동 사진 제외 전후 Brush PSNR** 로 간접 확인

## 오늘 밤 계획 (GPU 작업은 하나씩)
1. diagnose.py 로 원래 1b 의 순간이동 전체 목록 (CPU)
2. 같은 프레임(4차 1a 1100장 사본 → up\1a)으로 재정합: sequential(overlap 20, loop) + global → up\1b
3. 같은 매칭 DB 로 incremental → 비교용 폴더. 재투영 오차·등록률·순간이동 수 비교
4. Brush 기준선 설정(1920/300만/SH2/30,000, 평가 8장마다 1장)으로 원래 1b vs 새 정합 PSNR 비교
5. 나은 정합으로 2560 학습 (8GB 에서 메모리 부족이면 상한 낮춤) → tools/fill 바탕층·시작점 보강

## 아침 정리 (11-27 09:10)
- 지금 가장 나은 결과: **`F:\Sample\4차\up\1c_fill_origskip\fill_with_backdrop.ply`** (원래 정합 − 순간이동 25장, Brush 기준선, 바탕층 포함). PSNR 만 보면 `up\1c_brush_origskip\export_30000.ply` (25.75) 가 최고
- 정합을 새로 하는 것(global/incremental)은 효과 없음 → 원래 정합 유지
- 해상도 2560 은 효과 없음 (시간만 1.5배)
- 남은 핵심: 자세 미세 보정 + 노출 보정 (LichtFeld `--pose-opt`, `--ppisp`). **VS2022 Installer > 개별 구성 요소 > "Windows용 C++ Clang 컴파일러" 설치** 후 `F:\Sample\tools\lfs\build_lfs.bat` 다시 실행

## 사용자 확인 (11-27 오전) → 바탕층 "뚫린 곳만"
- 사용자 의견: `1c_brush_origskip` 은 괜찮지만 바닥 등 일부 뚫림, `fill_with_backdrop` 은 살짝 뭉개져 보임
- 원인 분석 (바탕층 점 ↔ 학습 스플랫(불투명도>0.3) 최근접 거리): 바닥은 92% 가 5cm 안에 학습 스플랫 있음(덮임), 천장은 94% 가 비어 있음,
  **벽 바탕층은 80% 가 실제 벽과 5cm 넘게 어긋남** → 반투명 벽 스플랫 뒤로 흐린 바탕층(1/4 크기 사진 6장 중간값 색)이 비쳐 뭉개짐
- 새 도구 `tools/fill/merge_holes.py`: 벽 바탕층은 빼고, 바닥·천장 바탕층 중 학습 스플랫이 덮지 않은 곳(반경 4.5cm 안 불투명도 합 < 1)만 합침
  - `up\1c_brush_origskip\holes_backdrop.ply` : 기준선(순간이동 제외) + 바탕층 3.8만 개 ← **추천 확인 대상**
  - `up\1c_fill_origskip\fill_holes.ply` : 빈 곳 채우기 학습 + 바탕층 0.9만 개

## 폴더 정리 (11-27 12:00)
- `up\1b` (global 정합, 못 씀) → `up\1b_global_skip` 로 이름 변경 (삭제 안 함, `1b` 이름은 새 파이프라인 실행용으로 비워 둠)
- `up\*.log`, `run_fill_origskip.ps1` → `up\logs\` 로 모음
- `F:\Sample\tools\lfs` 의 실패한 빌드 로그(build_log, build_log2, vs2026/경로 오류 시도) 삭제. 내용은 위 시행착오에 요약돼 있음
- LichtFeld 빌드 시행착오 추가: ① build_lfs.bat 의 한글 주석을 cmd 가 CP949 로 잘못 읽어 vcvars64 줄이 실행 안 됨 → bat 은 ASCII 만
  ② vcpkg 가 VS2026(18, MSVC 14.52) 컴파일러를 고름 → `F:\Sample\tools\lfs\triplets\x64-windows.cmake` 에 `VCPKG_VISUAL_STUDIO_PATH`(VS2022, 백슬래시 두 번) + `VCPKG_PLATFORM_TOOLSET v143`, cmake 에 `-DVCPKG_OVERLAY_TRIPLETS` → MSVC 14.44 로 빌드 진행 중

## LichtFeld 실험 (11-27 오후)
- 빌드 성공 14:07 (VS2022 고정, clang-cl). 마지막 걸림: uv 다운로드 후 `.part` → `.zip` 이름 바꾸기 실패(백신 검사 잠금 추정) → 체크섬 확인 후 직접 바꿈
- 빌드 중간물 vcpkg\buildtrees(16GB)·packages(4.3GB) 삭제 (바이너리 캐시 vcpkg_cache 에 남음)
- 새 스크립트 `3_train/lfs_eval.ps1` (학습+평가+통계), `tools/compare_eval.py` (같은 사진끼리 PSNR 비교, Brush/LichtFeld 둘 다)
- v0.5.4.3 기본값 주의: `--exposure-correction`(사진별 노출·WB + 국소 격자), 보정 왜곡(calibration correction, 1px 어긋남 흡수) 이 **기본으로 켜져 있음**
- L0 (pose-opt 없음): 25.62 / 0.863, 같은 133장에서 Brush 25.75 와 거의 같음. 23분 (Brush 36분보다 빠름)
  - **정렬 후 PSNR 27.14**: 렌더를 사진에 맞춰 옮기면 +1.5 → 평가 사진 렌더가 중앙 2.0px(최대 53.8px) 어긋나 있음 = **자세 오차가 실제로 있음**
  - 반투명 96.2% (Brush 86.9%) → VR 비침에는 더 나쁨. MCMC 특성으로 보임
- L1 (`--pose-opt`): **25.90 / 0.869**, 같은 133장 L0 대비 +0.28 (74% 사진 개선), 평가 렌더 어긋남 중앙 2.0 → 1.4px. 45분 (2배 느림)
  - **자세 보정이 효과 있음 = 복도 PSNR 의 원인 중 하나가 자세 오차였음** (QUALITY_PLAN 1단계 결론)
  - 평가 사진(8장마다 1장)은 학습에서 빠져 자세가 보정되지 않으므로 실제 개선은 더 클 수 있음
  - 반투명 96.4% 그대로 → 다음: L2 = L1 + `--strategy=mrnf` (v0.5.4.3 기본 전략, MCMC 대신)
- L2 (L1 + `--strategy=mrnf`): **26.51 / 0.879**, L1 대비 +0.60 (84% 사진 개선). **반투명 96.4 → 63.3%, 바늘형(>10배) 17.3 → 1.3%**. 37분
  - MRNF 는 v0.5.4.3 기본 `--exposure-correction` 이 켜지는데 `--ppisp` 와 같이 못 씀 (첫 시도 즉시 실패) → 한 가지만 바꾸려고 `--no-exposure-correction` 추가
  - **지금까지 최고 결과**: `F:\Sample\4차\up\1c_lfs_origskip_poseopt_mrnf\splat_30000.ply` → VR/SuperSplat 확인 대상
  - 다음: L3 = L2 에서 PPISP → `--exposure-correction`(사진별 노출·WB + 국소 격자, v0.5.4.3 기본) 으로 바꿈. `lfs_eval.ps1 -NoPpisp`
