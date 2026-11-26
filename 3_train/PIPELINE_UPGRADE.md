# 3DGS 파이프라인 업그레이드 레시피 (2026-11-25)

> Fable 조사(1차 자료 기준, 실제 설치·실행은 안 해 봄) + Claude가 LichtFeld CLI 플래그와 COLMAP 4.0 릴리스 노트를 다시 확인함. 시간·VRAM은 추정. 실행 전 각 명령 `--help`로 옵션명 확인할 것.
> 대상: 팀원 PC (RTX 5070 12GB, Windows), 무료 도구만, 복도 1 한 구간부터.

## 추천 스택

| 단계 | 추천 | 대안 |
|---|---|---|
| 정합 | **COLMAP 4.2.1**: LightGlue 매칭 + sequential/loop + 전역 매퍼(GLOMAP 통합) | 같은 DB로 incremental `mapper` |
| 학습 | **LichtFeld Studio v0.5.4.3** (소스 빌드 무료): MCMC + PPISP(노출·WB 보정) + 배경 변조 | **gsplat** `simple_trainer mcmc` (포즈 최적화·깊이 손실) |
| 기준선 | 현재 Brush 0.3 결과 (같은 구간·같은 평가 프레임) | — |

**지금 구성 대비 실제로 바뀌는 것**
1. 매칭 exhaustive(45시간) → sequential + loop → 프레임 솎아내기 불필요
2. 노출·WB 보정(PPISP) — Brush엔 없음
3. **배경 변조(`--bg-mode=modulation`, v0.5.3 전에는 `--bg-modulation`)**: 검은 배경이면 어두운 곳이 투명하게 남는데, 배경색을 바꿔 가며 학습해 불투명 스플랫으로 덮게 함 → 분홍 배경 비침(반투명 95%)에 직접 대응
4. 왜곡 보정을 학습기가 처리(`--undistort`) → `image_undistorter` 단계 없어짐

## 제외한 것과 이유
- MASt3R-SfM: 비상업 라이선스, Windows 안내 없음, 중소 규모용 → 구간당 2,000장엔 부적합
- VGGT: 수백 장 규모
- 3DGRUT: Windows·RTX 50 지원은 하지만 COLMAP 데이터에서 롤링셔터 사용법·PLY 호환이 문서에 없음 → Unity용으로 위험
- Postshot: 무료 플랜은 PLY 내보내기 없음
- Brush: 릴리스 v0.3.0에 머물러 있음, 노출 보정·깊이 없음
- 롤링셔터·모션블러를 직접 모델링하는 검증된 무료 경로는 없음 → 프레임 선별 + 포즈 최적화로 완화, 근본 해결은 재촬영

## 설치 (Windows + RTX 5070)
공통: NVIDIA 드라이버 570 이상.

**COLMAP**: GitHub 릴리스의 Windows CUDA 빌드 4.2.1

**LichtFeld Studio** (위키 Build Instructions – Windows)
1. Visual Studio **2022** Community + "Desktop development with C++" (2019·2026은 안 됨)
2. CUDA Toolkit **12.8** (다른 버전 먼저 제거), Git, Strawberry Perl 5.42+, CMake 3.30+
3. "x64 Native Tools Command Prompt for VS 2022"에서, 공백 없는 경로에:
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && .\bootstrap-vcpkg.bat -disableMetrics && cd ..
git clone --recurse-submodules https://github.com/MrNeRF/LichtFeld-Studio
cd LichtFeld-Studio
git checkout v0.5.4.3   # --pose-opt(자세 최적화) 가 들어간 버전 (2026-11-23)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build -j
```
- 한글 Windows는 VS 영어 언어팩 필요, 백신·OneDrive가 vcpkg를 막을 수 있음, 메모리 부족하면 `-j4`

**gsplat (대안)**: Python 3.11+, PyTorch 2.7+ CUDA 12.8 빌드(pytorch.org 선택기에서 명령 복사) → VS Build Tools `vcvars64.bat` → `git clone --recursive https://github.com/nerfstudio-project/gsplat.git` → `pip install .` → `pip install -r examples/requirements.txt --no-build-isolation`. 5070은 소스 빌드가 안전

## 복도 1 실행 예

**① 프레임**: `pick_best.py --min-diff 4`로 촘촘히, 솎아내기 끔(`$ThinAbove` 크게). 파일명 순서 = 촬영 순서여야 함

**② 정합**
```
colmap feature_extractor --database_path db.db --image_path images ^
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV
colmap sequential_matcher --database_path db.db ^
  --FeatureMatching.type SIFT_LIGHTGLUE ^
  --SequentialMatching.overlap 20 --SequentialMatching.loop_detection 1 ^
  --SequentialMatching.loop_detection_period 10 ^
  --SequentialMatching.loop_detection_num_images 50
copy db.db db_global.db
colmap view_graph_calibrator --database_path db_global.db
colmap global_mapper --database_path db_global.db --image_path images --output_path sparse
```
- `view_graph_calibrator`는 DB를 직접 고쳐서 사본에 씀. 영상 프레임은 EXIF가 없어 필요
- loop detection용 vocab tree는 자동 다운로드. 옵션명은 `colmap sequential_matcher --help`로 확인
- LightGlue가 안 되면(ONNX 미포함 등) SIFT 기본 매칭 + sequential/loop만으로도 시간 문제는 해결
- 등록 장수가 적거나 모델이 쪼개지면 같은 `db.db`로 `colmap mapper`(incremental)
- 예상(추정): 2,000장 매칭 수십 분, 전역 매핑 수 분~수십 분

**③ 학습 (LichtFeld)** — 폴더에 `images/`, `sparse/0/`만 있으면 됨
```
LichtFeld-Studio.exe -d D:\up\1 -o D:\up\1_lfs --strategy=mcmc --max-cap=2000000 --iter=30000 ^
  --sh-degree=2 --max-width=2560 --undistort --ppisp --bg-mode=modulation --export=ply ^
  --eval --test-every=8 --headless
```
- `--enable-mip`는 **끔** (Unity 셰이더는 고정 0.3px 확장만 써서 필터가 어긋남. 셰이더 고친 뒤 켜기)
- `--gut`(3DGUT)도 끔 (Unity 일반 래스터라이저와 차이 미검증)
- VRAM 부족: `--max-width=1920`, `--max-cap` 낮추기 (`--tile-mode` 는 v0.5.3 부터 없음)
- 개수 줄일 땐 사후 lite 대신 `--enable-sparsity`
- 예상(추정): 30k 스텝 20~40분, 12GB 이내

**③′ 학습 (gsplat 대안)**
```
python examples/simple_trainer.py mcmc --data_dir D:\up\1 --data_factor 2 --result_dir D:\up\1_gs ^
  --strategy.cap-max 2000000 --sh_degree 2 --pose_opt --depth_loss --random_bkgd ^
  --post_processing ppisp --test_every 8 --save_ply --ply_steps 30000
```
- `data_factor 2`는 `images_2` 폴더가 미리 있어야 함. 4K 그대로(`1`)는 12GB에서 위험
- `antialiased`는 같은 이유로 끔

**④ 정리**: `clean_section.py`는 그대로, 불투명 바탕층과 `lite.py` 절반 삭제는 빼고 비교

## 비교 방법 (복도 1, 스플랫 200만으로 맞춤, 평가 프레임 8장마다 1장)
1. 평가 프레임 PSNR/SSIM/LPIPS (LichtFeld `--eval`, gsplat 자체 평가)
2. `tools/ply_stats.py` 반투명(<0.5) 비율, 바늘형 비율 (기준: 95%)
3. SuperSplat 분홍 배경, 고정 시점 4곳(정면, 45°, 벽에 붙은 70°, 천장)
4. Unity 메뉴 9 방식 임포트, 같은 시점 스크린샷

## 함정
- MCMC가 반투명 비율을 낮춘다는 보장은 없음 → 비침은 분홍 배경 테스트로 판단. 주 해결책은 배경 변조·정합·노출 개선
- 예전 "보강 시작점 + gsplat MCMC" 역효과 기록(960폭) → 시작점 보강 유무 각각 돌려 볼 것
- PPISP를 켜면 PLY 색이 보정 기준 색 공간 → 구간 간 밝기 차이는 줄 수 있으나 눈으로 확인
- **재정합하면 좌표계가 바뀜** → `place.py`/`layout.json` 구간 배치 변환을 다시 구해야 함 (가장 큰 추가 비용)
- 전역 매퍼는 중단 후 이어하기 안 됨
- 깊이 prior: gsplat `depth_loss`는 COLMAP 희소점 깊이만 씀 → 점 없는 천장·흰 벽엔 효과 제한. 단안 깊이(Depth Anything V2 등)는 코드 작업 필요, 2단계로

## 출처
- COLMAP 4.0.0 릴리스: https://github.com/colmap/colmap/releases/tag/4.0.0 , 릴리스 목록: https://github.com/colmap/colmap/releases
- COLMAP 특징·CLI·튜토리얼: https://colmap.github.io/features.html , https://colmap.github.io/cli.html , https://colmap.github.io/tutorial.html
- LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio , CLI: https://github.com/MrNeRF/LichtFeld-Studio/wiki/Command-Line-Options , 기능: https://github.com/MrNeRF/LichtFeld-Studio/wiki/Feature-Documentation , Windows 빌드: https://github.com/MrNeRF/LichtFeld-Studio/wiki
- gsplat: https://github.com/nerfstudio-project/gsplat , Windows 설치: https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md
- PyTorch: https://pytorch.org/get-started/locally/
- Brush: https://github.com/ArthurBrussee/brush/releases
- 3DGRUT: https://github.com/nv-tlabs/3dgrut
- MASt3R: https://github.com/naver/mast3r , VGGT: https://arxiv.org/abs/2503.11651 , 3DGS-MCMC: https://arxiv.org/abs/2404.09591
- Postshot 가격: https://www.jawset.com/shop/pricing
