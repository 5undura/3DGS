# NoClip 3DGS 파이프라인

중앙대 학생회관 지하(흰 벽, 어두운 복도·방 6구간)를 Osmo Pocket 3 4K 영상으로 찍어
프레임 추출 → COLMAP 정합 → 학습(LichtFeld / Brush) → PLY 로 만드는 도구 저장소.
결과 PLY 는 Unity VR 게임(AVT 저장소)에서 씀. 목표: 반투명·비침이 적고 선명한 스플랫.

## 경로
- 원본 데이터: F:\Sample\4차 (0_영상, 1_프레임, 2_정합데이터, 3_학습결과) — 읽기 전용으로 취급
- 새 실험 작업 폴더: F:\Sample\4차\up ($Work). 구조: Na(프레임), Nb(정합), Nc_<학습기>(PLY). 영상(N.MP4)은 $Work 또는 $VideoDir
- 예전 구간 배치(건물 좌표 변환): tools/layout_v4.json (AVT 저장소 Tools/splat_processing/v4/layout.json 사본)
- fill 도구: F:\Sample\tools\fill (저장소 tools/fill 과 같은 내용)
- GPU: 이 PC RTX 3060 Ti 8GB ($GpuProfile="3060ti"). 팀원 PC RTX 5070 12GB
- 설정은 config.ps1 (git 에 안 올라감). 안내서: 3_train/PIPELINE_UPGRADE.md, 3_train/LICHTFELD_SETUP.md

## 규칙
- 답변은 한국어로 짧게, 결론부터.
- 커밋 작성자는 5undura. 커밋 메시지에 Co-Authored-By, Claude 서명·세션 링크 줄을 넣지 않는다.
  커밋·푸시는 내가 시킬 때만.
- 원본 데이터(영상, 프레임, 정합 결과, PLY)는 절대 삭제·덮어쓰기 금지.
  빼야 하면 *_skip 폴더로 옮기고, 새 결과는 새 폴더에 만든다.
  예외: 방금 실패한 실행이 만든 중간 산출물.
- F:\Sample\4차 의 기존 폴더 안에서 파이프라인을 돌리지 않는다. 항상 $Work(up) 에서.
- 10분 넘는 작업은 백그라운드로 실행하고 로그 파일에 남긴 뒤, 로그를 주기적으로 확인해 보고한다.
- 잘 되던 Brush 설정(1920, 300만, SH2, 30,000)은 기준선으로 보존. 바꿀 때는 기준선 결과를 남긴다.
- 실험은 한 번에 한 가지만 바꾼다. 모든 실험은 3_train/README.md 표에 기록
  (날짜, 구간, 바꾼 것, 스플랫 수, 반투명 비율, PSNR, 시간, 결과 경로).
- 시스템 소프트웨어 설치(Visual Studio, CUDA, 드라이버, pip 전역 설치 등)는 먼저 물어본다.
- 돈 드는 서비스 금지. 옵션 이름은 추측하지 말고 --help 출력으로 확인한다.
