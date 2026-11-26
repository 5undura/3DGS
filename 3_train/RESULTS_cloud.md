# 클라우드 작업 정리 (2026-11-26, GPU 없이 코드만, 실행 전)
**한 일**
- PS 스크립트 점검·수정: 끊긴 실행 흔적(Na_tmp, db/sparse/data, 실패한 Nc)을 `*_failed_<시각>` 으로 옮기고 다시 시작(삭제 안 함), undistort 결과를 `data\sparse\0` 으로 옮김(전에는 `sparse\` 바로 아래라 학습 단계가 모두 건너뛰었을 것), model_analyzer 출력 파싱, 학습기 `| Out-Host`(끝날 때까지 대기), config.ps1 없으면 중단
- `pick_best.py --abs-floor-pct P`: 영상 전체 선명도 하위 P% 버림(강제 저장 포함), 빈 구간 출력. 기본 0 은 예전과 같은 결과(합성 영상에서 바이트 단위로 같음)
- `tools/realign_layout.py`(재정합 후 layout_up.json), `tools/compare_runs.py`(PLY 지표 표·마크다운). 가짜 데이터로 검증
**Windows 에서 먼저 확인**
1. 버전: `COLMAP.bat -h`(4.x), `LichtFeld-Studio.exe --help`, `brush_app.exe --help`, `python -c "import cv2;print(cv2.__version__)"`, PowerShell `$PSVersionTable`
2. COLMAP 옵션 이름: `COLMAP.bat sequential_matcher -h`(SequentialMatching.overlap / loop_detection), `view_graph_calibrator -h`, `global_mapper -h`, `feature_extractor -h`(ImageReader.single_camera / camera_model). loop detection 용 vocab tree 는 처음에 인터넷으로 내려받음
3. LichtFeld: --strategy --max-cap --iter --sh-degree --max-width --ppisp --bg-mode --export --eval --test-every --headless --pose-opt 가 --help 에 있는지 (v0.5.4.3 소스로는 확인됨, --bg-modulation/--tile-mode 는 없어짐)
4. Brush: $Iterations 를 30000 이 아닌 값으로 바꾸면 총 스텝 옵션(--help 에서 이름 확인)도 넘겨야 함. 지금은 --export-every 에만 씀
5. 한글 경로(F:\Sample\4차\up)를 COLMAP 이 읽는지. 안 되면 영문 경로 작업 폴더로
**60초 시험 실행**: `ffmpeg -ss 0 -t 60 -i <영상>\1.MP4 -c copy F:\Sample\4차\up_test\1.MP4` → config.ps1 을 `$Work="F:\Sample\4차\up_test"`, `$Sections=1`, `$Iterations=2000` 으로 → `.\run_all.ps1` → `up_test\night_log.txt`(UTF-16), `1b\data\sparse\0`, `1c_lichtfeld\*.ply` 확인
- 하한 시험: `python 1_extract\pick_best.py 1.MP4 test_floor --abs-floor-pct 10` (2번 읽어서 약 2배 시간)
- 배치: `python tools\realign_layout.py --section 1 --old <예전 1b\data> --new up\1b\data` → 짝 수·잔차(cm) 확인, SuperSplat/Unity 로 눈 확인
- 비교: `python tools\compare_runs.py 기준=<Brush PLY> LFS=<새 PLY> --md tmp.md` → 3_train/README.md 표에 옮김
