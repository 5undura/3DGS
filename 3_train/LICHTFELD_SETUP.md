# LichtFeld Studio 설치 (Windows, 무료 소스 빌드)

LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio (GPLv3). Windows 실행 파일은 기부제 배포, 소스 빌드는 무료.
빌드 방법 원문: https://github.com/MrNeRF/LichtFeld-Studio/wiki (Build Instructions – Windows)

## 준비물
- NVIDIA 드라이버 570 이상
- Visual Studio **2022** Community + "Desktop development with C++" (2019·2026 안 됨)
  - **개별 구성 요소 "C++ Clang Compiler for Windows"(clang-cl) 도 필요** (v0.5.4.3 의 libplacebo 의존성. 없으면 vcpkg 단계에서 실패, 2026-11-27 확인)
  - 한글 Windows면 Visual Studio Installer에서 **영어 언어팩** 추가
- CUDA Toolkit **12.8** (다른 버전이 깔려 있으면 먼저 제거)
- Git, CMake 3.30+, Strawberry Perl 5.42+

## 빌드
"x64 Native Tools Command Prompt for VS 2022"를 열고, **공백 없는 경로**(예: `C:\tools`)에서:
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && .\bootstrap-vcpkg.bat -disableMetrics && cd ..
git clone --recurse-submodules https://github.com/MrNeRF/LichtFeld-Studio
cd LichtFeld-Studio
git checkout v0.5.4.3   # --pose-opt(자세 최적화) 가 들어간 버전 (2026-11-23)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_TOOLCHAIN_FILE="../vcpkg/scripts/buildsystems/vcpkg.cmake" -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```
- `-DCMAKE_CUDA_ARCHITECTURES=native`: 빌드하는 PC의 그래픽카드용으로만 만듦 (3060 Ti = 86, 5070 = 120). **빌드한 파일을 다른 세대 그래픽카드 PC로 옮겨 쓰려면** `"86;120"`처럼 둘 다 적기. 이 옵션이 위키 방식과 충돌하면 빼고 빌드
- 빌드 중 메모리 부족이면 `cmake --build build -j4`
- 백신·OneDrive가 vcpkg 폴더를 막으면 예외 처리
- 완성된 실행 파일 경로를 `config.ps1`의 `$LichtFeld`에 적기
- 실행 전 `LichtFeld-Studio.exe --help`로 옵션 이름 확인 (`3_train/train_all.ps1`에서 쓰는 옵션: `--strategy=mcmc --max-cap --iter --sh-degree --max-width --ppisp --bg-mode=modulation --export=ply --eval --test-every --headless`, v0.5.4.3 소스로 확인. `--bg-modulation`·`--tile-mode` 는 v0.5.3 부터 없음, v0.5.4 부터 `--export=ply` 없으면 PLY 안 나옴. 자세 최적화 `--pose-opt` 는 v0.5.4.3 부터)

## COLMAP 4.x
- https://github.com/colmap/colmap/releases 에서 Windows CUDA 빌드(4.2.1 이상) 받기. GLOMAP 전역 정합이 `global_mapper`로 들어 있어 따로 설치할 것 없음
- 50 시리즈(5070 등)는 옛 COLMAP CUDA 빌드가 안 돌 수 있음 → 최신 버전
- 처음 한 번 `COLMAP.bat sequential_matcher -h`로 `config.ps1`의 `$MatcherArgs` 옵션 이름이 맞는지 확인
