# 이 파일을 복사해서 config.ps1 로 이름을 바꾸고, 자기 PC 경로로 고쳐서 사용.
# config.ps1 은 .gitignore 에 들어 있어서 저장소에 올라가지 않음.

# COLMAP (CUDA 빌드) 실행 파일
$Colmap = "C:\tools\colmap\COLMAP.bat"

# Python (opencv-python, numpy 설치된 것)
$Python = "python"

# 작업 폴더: 촬영 영상 1.MP4, 2.MP4 ... 를 넣어 둔 폴더
#   결과는 같은 폴더 안에 1a(프레임) / 1b(정합) / 1c(PLY) 로 생김
$Work = "D:\capture\6"

# 처리할 구간 번호
$Sections = 1..6

# 프레임이 이 장수보다 많으면 솎아냄 (정합 시간 절약)
$ThinAbove = 1200
# 솎아낸 뒤 목표 장수 (대략)
$ThinTarget = 1100

# 매칭 방식: exhaustive(느리지만 안정적, 구간당 수 시간) / sequential(빠름, 영상 순서대로만 비교)
$Matcher = "exhaustive"
