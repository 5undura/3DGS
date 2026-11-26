# 이 파일을 복사해서 config.ps1 로 이름을 바꾸고, 자기 PC 경로로 고쳐서 사용.
# config.ps1 은 .gitignore 에 들어 있어서 저장소에 올라가지 않음.
# 한 번에 돌리기: run_all.bat 더블클릭 (또는 PowerShell 에서 .\run_all.ps1)

# ---------------- 경로 ----------------
# COLMAP 4.x (Windows CUDA 빌드). 4.0부터 GLOMAP 전역 정합(global_mapper)이 들어 있음
$Colmap = "C:\tools\colmap\COLMAP.bat"
# Python (opencv-python, numpy 설치된 것)
$Python = "python"
# 학습기 실행 파일 (쓰는 것만 맞으면 됨)
$LichtFeld = "C:\tools\LichtFeld-Studio\build\LichtFeld-Studio.exe"
$Brush     = "C:\tools\brush\brush_app.exe"

# 작업 폴더: 촬영 영상 1.MP4, 2.MP4 ... 를 넣어 둔 폴더
#   결과는 같은 폴더 안에 1a(프레임) / 1b(정합) / 1c_<학습기>(PLY) 로 생김
#   ⚠️ 새 실험은 항상 "새 폴더"에서: 이미 Na(프레임)·Nb\data(정합)가 있는 구간은 건너뛰므로,
#      예전 작업 폴더에서 돌리면 아무 것도 새로 안 함. 예: $Work = "F:\Sample\4차\up", 영상은 $VideoDir 로 지정
$Work = "D:\capture\6"
# 영상이 $Work 가 아닌 다른 폴더에 있으면 지정 (비워 두면 $Work 에서 찾음)
$VideoDir = ""

# 처리할 구간 번호 (처음엔 1 하나로 시험)
$Sections = 1..6

# ---------------- 1. 프레임 추출 ----------------
# pick_best.py 옵션. --min-diff 를 낮출수록 많이 뽑음(기본 6). --sharp-ratio 를 올릴수록 흐린 프레임을 엄격하게 거름(기본 0.75)
# --abs-floor-pct P: 영상 전체 선명도 하위 P% 프레임은 버림 (기본 0=끔, 켜면 영상을 2번 읽음). 예: @(..., "--abs-floor-pct", "10")
$ExtractArgs = @("--min-diff", "4", "--sharp-ratio", "0.8")

# ---------------- 2. 정합 ----------------
# 매칭: sequential(영상 순서대로 + 같은 곳 다시 지날 때 루프 연결, 빠름) / exhaustive(모든 쌍, 2,000장이면 수십 시간)
$Matcher = "sequential"
# 매칭 추가 옵션. COLMAP 버전마다 이름이 다를 수 있으니 처음 한 번 `COLMAP.bat sequential_matcher -h` 로 확인
$MatcherArgs = @("--SequentialMatching.overlap", "20", "--SequentialMatching.loop_detection", "1")
# 정합 방식: global(GLOMAP, 빠름) / incremental(기존 mapper, 느리지만 안정적). global 이 실패하면 자동으로 incremental 로 다시 함
$Mapper = "global"
# 프레임이 이 장수보다 많으면 솎아냄. sequential 이면 솎을 필요 없음 -> 아주 크게
$ThinAbove = 99999
$ThinTarget = 1100

# ---------------- 3. 학습 ----------------
# 학습기: lichtfeld / brush
$Trainer = "lichtfeld"
# 그래픽카드 프로필: 3060ti (8GB) / 5070 (12GB). 학습 해상도·스플랫 상한이 달라짐
$GpuProfile = "3060ti"
$Iterations = 30000
$ShDegree = 2
