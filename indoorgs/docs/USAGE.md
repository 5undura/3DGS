# IndoorGS 사용법 (Windows)

**요약**: `windows\install.bat` 한 번 → `windows\new_project.bat` 로 프로젝트 만들기 → `project.yaml` 에 영상 적기 → `windows\IndoorGS.bat` 에서 공간·동선 → **전체 실행** → 결과 화면.
터미널 없이 더블클릭만으로 됩니다. 아직 실제 촬영 데이터로 처음부터 끝까지 돌려 보지 않았습니다(→ [PC_CHECKLIST.md](PC_CHECKLIST.md)).

## 1. 설치

먼저 있어야 하는 것 (직접 설치. IndoorGS 는 시스템에 아무것도 설치하지 않음):
- Python 3.10 이상 (설치할 때 "Add python.exe to PATH" 체크)
- COLMAP 4.2.1 이상 Windows CUDA 빌드 (`COLMAP.bat`)
- Brush 0.3.0 (`brush_app.exe`) — 기본 학습기
- (선택) LichtFeld Studio v0.5.4.3 — 고화질. 빌드 방법: [3_train/LICHTFELD_SETUP.md](../../3_train/LICHTFELD_SETUP.md)

그다음 `indoorgs\windows\install.bat` 더블클릭.
- 저장소 폴더에 `.venv` (이 도구 전용 Python 환경)를 만들고 IndoorGS 와 필요한 패키지(GUI, DXF, scipy)를 그 안에만 설치
- 다시 실행하면 `.venv` 는 그대로 두고 패키지만 갱신. 꼬이면 `.venv` 폴더를 지우고 다시 실행
- `.venv` 는 git 에 안 올라감 (GitHub Desktop 변경 목록에도 안 보임)

## 2. 새 프로젝트

`windows\new_project.bat` 더블클릭 → 이름 입력 (예: `student_hall`).
- 이름만 쓰면 `%USERPROFILE%\IndoorGS_projects\<이름>`, 전체 경로(`D:\scan\student_hall`)도 됨. **영문 경로 권장**
- COLMAP·Brush·LichtFeld 를 자동으로 찾아 `project.yaml` 에 적음. 찾는 곳: PATH, `C:\tools`, `%LOCALAPPDATA%`·`Program Files` 의 도구 이름 폴더, bat 맨 위 `SEARCH` 폴더(기본 `F:\Sample\tools`, 깊이 4). 여러 개면 가장 최근 파일
- 도구가 다른 곳에 있으면 메모장으로 `new_project.bat` 을 열어 맨 위 `SEARCH`·`PROJECTS` 를 고치면 됨

## 3. 영상 넣기

`new_project.bat` 이 메모장으로 `project.yaml` 을 엽니다. `sections:` 아래에 구간마다 이름과 영상:

```yaml
sections:
  - name: corridor1          # 영문·숫자·_·- (폴더 이름이 됨)
    video: F:/Sample/4차/up/1.MP4
  - name: room_b117
    video: F:/Sample/4차/up/2.MP4
train: {trainer: brush, preset: default}     # 고화질: trainer: lichtfeld
```
저장하고 메모장을 닫으면 `indoorgs check` 결과가 나옵니다 (도구·영상 있음/없음, 자동으로 찾은 다른 후보).
GUI 의 **프로젝트** 화면에서도 구간·영상이 맞는지 볼 수 있습니다.

학습 프리셋:
- Brush `default` = 기준선(1920, 300만, SH2, 30,000회, 복도 1 약 36분), `fast` = 점검용 5,000회
- LichtFeld `default` = 복도 1 최고 설정(MRNF + 자세 보정, 80,000회, 약 91분), `quick` = 30,000회(약 42분), `fast` = 점검용

## 4. 화면 열기

`windows\IndoorGS.bat` 위에 프로젝트 폴더를 끌어다 놓기 (또는 더블클릭 후 폴더를 창에 끌어다 놓고 Enter).
브라우저에 로컬 화면(`http://127.0.0.1:8765/`)이 열립니다. 끄려면 검은 창을 닫기.

## 5. 공간 만들기 (선택, 권장)

**공간** 화면. 공간이 있으면 정합을 도면 좌표에 맞추고, 구간 합치기도 변환 없이 됩니다.
- **방 추가** → 끌어서 이동·크기·회전, **문 추가** → 벽 클릭. 단위는 미터
- 도면 사진이 있으면 `floorplan\plan.png` 로 넣어 배경으로 깔고 그 위에 박스를 맞춤
- **도면 불러오기**: DXF·SVG 파일이나 도면 사진(네 모서리 클릭 + 실제 가로·세로 길이)에서 방 초안을 만들어 겹쳐 보여 줌 → 바꾸기 / 추가하기 / 취소. 사진은 완전 자동이 아니라 초안이니 박스로 다듬기
- 사진(비상대피도 등)은 복도·방을 영역으로 나눠 박스 초안을 만듦: 복도(청록, ㄱ·T 자는 박스 여러 개)·방(보라)·믿음 %(60% 미만 주황)·문.
  초안 방을 클릭해 **삭제**(Delete) / **촬영 안 함** 표시 / **복도↔방** 바꾸기, **믿음 낮은 것 지우기**.
  빠진 방은 **클릭해서 영역 추가**(벽 안쪽을 채움, 벽이 끊긴 곳은 직접 그리기). 촬영 안 한 방은 남겨도 됨(벽 단계는 카메라 있는 방만 씀).
  축척은 두 점 실측으로 확인 (대피도의 20m 같은 거리 글자는 경로 길이라 참고만). 예전 방식은 설정의 "예전 방식 (이진화)"
- 오른쪽 **검사** 목록: 방 겹침, 벽 위에 없는 문 등. 누르면 그 방·문이 선택됨
- **저장** → `floorplan\plan.json`

## 6. 동선 그리기 (선택)

**동선** 화면 → 구간 고르기 → **브러시**로 걸은 길을 그림 → 영상을 보며 모퉁이·문에서 **고정점**(그 시각의 위치) 찍기 → **저장**.
공간과 동선이 둘 다 있으면 프레임별 예상 위치를 COLMAP 에 함께 줘서 흰 벽·어두운 복도에서 정합이 덜 틀어집니다.
영상이 화면에서 재생되지 않으면 아래 문제 해결 참고.

## 7. 실행

**실행** 화면 → **전체 실행** (또는 구간별, "~까지"로 중간에서 멈추기).
순서: 프레임 추출 → (동선 위치) → 정합 → 진단(순간이동 사진 빼기) → (벽 면 맞추기·초기 점 심기) → 학습 → (벽에 붙이기) → 정리 → 보강(바닥·천장 구멍) → 짧은 재학습 → 합치기 → 보고서.
- 괄호 단계는 공간 + 동선이 있어 도면 좌표로 정합됐을 때만 돎. 벽 평탄화는 `project.yaml` 의 `walls:` 로 끄고 켬.
  **기본값 (11-28, `experiments/wall_ab/RESULTS2.md`)**: `enabled`·`seed` 켬 (면 맞추기 + 면 위 초기 점 = 확실한 이득),
  `repair`·`remove_behind` 는 **끔**. `snap`·`flatten`(벽에 붙이고 납작하게)은 안 적으면 **짧은 재학습(finetune)이 켜져 있고 학습기가 Brush 이고 seed 가 켜졌을 때만 켬**
  (학습된 스플랫을 고치기만 하면 PSNR 이 떨어지지만, 고친 뒤 재학습하면 raw + 같은 재학습보다 나음: RESULTS2 E). 직접 `walls: {snap: false}` 처럼 적으면 그 값을 씀.
  `repair: true` 는 snap 필요. 원본 `splat.ply` 는 그대로 두고 `splat_walls.ply` 를 따로 만듦.
  `trust_plan_dims: true` (기본 끔): 줄자 도면 치수를 믿을 때만 — 점이 없어 못 맞춘 벽을 맞춘 평행 벽 + 도면 폭으로 추정
  `scale_search` (기본 켬): 마주 보는 맞춘 면 쌍으로 모델 축척/도면 축척을 재서 (2% 넘게 다르면 로그·보고서에 **축척 경고**, `walls\planes.json` `scale_check`)
  그 축척으로 먼 끝 벽을 먼저 옮긴 뒤 거리에 비례해 넓혀(최대 1.5 m) 다시 맞춤 (`source: "fit_scaled"`). 이 면에는 기본으로 초기 점을 안 심음 (`seed_scaled: true` 로 켬)
- 진단: 직접 맞춘 정합이면 "동선에서 멂" 문턱 = 맞춤 잔차 중앙값 x 5 (1~2 m, `diagnose.json` `far_from_prior_threshold`).
  모델이 뒤집혀 있으면(카메라 위 방향이 아래) 진단이 **실패**하고 `data_clean` 을 만들지 않음 → `register: {pair_verify: true}` 로 다시 정합
- 학습 후 정리(`cleanup`, snap 뒤·repair 앞)는 모든 구간에서 돎. 기본은 방 안 부유물 지우기(`floaters`)만 →
  `splat_clean.ply`, 지운 것은 `removed_clean.ply` 로 열어 확인. 다른 단계(아주 큰 스플랫 `huge`, 바늘형 쪼개기 `needles`, SH 색 튐 `sh`)는
  `project.yaml` 의 `cleanup: {steps: [floaters, huge, needles, sh]}` 처럼 넣어서 켬. 바늘 쪼개기는 스플랫 수가 늘어남 (합성 장면 바늘 11~16% 에서 +77~112%)
- 짧은 재학습(`finetune`, 보강 뒤·합치기 앞, 기본 켬, Brush 만): 후처리한 PLY 에서 원래 반복 번호부터 1000 반복 더 학습
  (늘리기 없음, 학습률 고정) → `train\<run>\finetune\splat_ft.ply`. 합치기는 이것을 씀 (입력·설정이 바뀌면 다시 재학습).
  `project.yaml`: `finetune: {enabled: true, iters: 1000, lr_mean: 1.0e-6, lr_scale: 6.0e-3, refine: false}`. 끄려면 `enabled: false`.
  LichtFeld 는 밀도 조절을 끄는 확인된 옵션이 없어 건너뜀. **PC 에서 `brush_app.exe --help` 에 `--start-iter`, `--lr-mean`, `--lr-mean-end`,
  `--lr-scale`, `--lr-scale-end`, `--growth-stop-iter`, `--refine-every` 가 있는지 확인** (없으면 이 단계가 멈추고 알려 줌).
- 정합(공간 + 동선이 있을 때): **두 번 정합 + 동선 자세로 짝 검증**이 기본 (`register: {pair_verify: true}`).
  앞뒤 사진 짝만으로 1차 정합 → 동선 12 m 안의 후보 짝 추가 → 1차 자세와 안 맞는 짝(복도 양 끝이 비슷해 생기는 가짜 짝)을 DB 사본에서 지움 → 2차 정합 → 수평 맞추고 도면에 맞춤.
  합성 복도에서 접힘·순간이동이 없어짐 (`experiments/prior_matching/RESULTS.md`). 정합 시간은 대략 두 배.
  손으로 기울여 찍은 영상이면 `gimbal_level: false`. 예전 한 번 정합으로 돌리려면 `pair_verify: false`.
  후보 범위: `verify_max_distance`(12 m, 매칭이 너무 오래 걸리면 8), `verify_max_neighbors`(300). 6 m·80장은 부족했음
  속도: 1차는 기본 3장마다만 정합하고 사이는 보간 (`pass1_stride: 3`, 예전처럼 전부면 1). 합성 복도에서 더 빠르고 접힘도 줄었음 (`experiments/e2e_v4/SPEED.md`).
  `pass2_mapper: global` 은 2차가 4~5배 빠르지만 **실험용, 위험** (seed 에 따라 크게 접힘). 쓰면 diagnose 결과(순간이동·뺀 사진)를 꼭 확인
  `mode: anchored` (실험용): 그린 동선을 강한 자세 정보로 써서 incremental 정합 없이 자세를 먼저 정하고(짐벌 수평 가정), 동선 근처 짝만 매칭 → 자세 고정 삼각측량 → 한 번 마무리.
  합성 복도에서 정합이 크게 빨라짐 (`experiments/anchored/RESULTS.md`). 기본값은 아직 바꾸지 않음. 쓰면 diagnose 결과를 꼭 확인. 손으로 기울여 찍은 영상에는 쓰지 말 것
- 끝난 단계는 다시 안 돌림. 중간에 꺼도 다음에 이어서 실행
- 로그는 화면 아래, 파일은 `logs\<구간>_<단계>.log`
- 학습 시간(3060 Ti, 복도 1): Brush default 약 36분, LichtFeld default 약 1.5시간. 추출·정합은 따로

## 8. 결과 확인

- **결과** 화면 = `report.html`: 구간별 프레임 수, 정합 장수, PSNR, 반투명 비율, 도면 위 카메라 위치
- 구간 PLY: `sections\<구간>\train\<학습기>_<프리셋>\splat.ply` (벽에 붙인 것 `splat_walls.ply`, 정리한 것 `splat_clean.ply`, 메운 것 `splat_repaired.ply`), 구멍 메운 것 `sections\<구간>\fill\holes_backdrop.ply`, 재학습한 것 `...\train\<학습기>_<프리셋>\finetune\splat_ft.ply` (합치기가 쓰는 최종)
- 건물 하나: `merged\building.ply`
- 보기: [SuperSplat](https://superspl.at/editor) 에 PLY 끌어다 놓기 (브라우저, 무료). VR 은 Unity(AVT 저장소)에서

## 문제 해결

- **한글 경로**: COLMAP 은 한글 경로에서 됨(확인함). 그래도 예전에 OpenCV·일부 도구가 조용히 실패한 적이 있어 **프로젝트는 영문 경로** 권장
- **COLMAP 옵션 오류**: `indoorgs check` 가 버전과 `pose_prior_mapper` 유무를 보여 줌. 4.2.1 이 아니면 옵션 이름이 다를 수 있음 → 최신 4.x 로. 오프라인이면 sequential 매칭의 loop_detection 이 어휘 트리를 못 받아 실패할 수 있음 → `project.yaml` 의 `register: {loop_detection: false}`
- **LichtFeld 없음/빌드 실패**: [LICHTFELD_SETUP.md](../../3_train/LICHTFELD_SETUP.md). VS **2022** + "C++ Clang Compiler for Windows", CUDA 12.8. 빌드용 bat 에 한글 금지. 3060 Ti 에서 빌드한 exe 는 5070 PC 에서 안 돌 수 있음(`"86;120"` 으로 다시 빌드). Visual Studio·CUDA 설치는 직접 결정
- **Brush 반투명 경고**: 보고서에서 반투명 비율 80% 이상이면 VR 에서 벽이 비칠 수 있음 (Brush 기준선 복도 1 = 86.9%). LichtFeld `default` 는 49.4% → 가능하면 LichtFeld
- **실패한 단계**: 출력 폴더가 `<이름>_failed_<시각>` 으로 옆에 치워지고 원본은 그대로. 로그를 보고 고친 뒤 다시 실행. 실패 폴더는 확인 후 직접 지워도 됨
- **영상이 화면에서 재생 안 됨**: Osmo Pocket 3 의 HEVC(10bit)는 브라우저가 못 틀 수 있음. 처리에는 문제없고 동선 화면 미리보기만 안 됨. 미리보기용 H.264 사본:
  (`F:\Sample\tools\ffmpeg_tool` 안의 ffmpeg.exe) `ffmpeg.exe -i 1.MP4 -vf scale=1920:-2 -c:v libx264 -crf 23 -an 1_preview.mp4` (원본은 그대로 두고 새 파일)
- **화면이 안 열림**: 검은 창의 주소를 브라우저에 직접 입력. 포트가 쓰이고 있으면 다른 IndoorGS 창을 닫기

## 촬영 팁 (복도 1 진단 결론)

- **천천히** 걷기. 빠르면 자세가 튀는 사진(순간이동)이 생김 (복도 1: 25장)
- **빠른 회전 피하기**. 방향 바꿀 때는 걸음을 멈추고 천천히 돌기
- **모퉁이·문 앞에서 2초 멈춤**, 모퉁이는 겹치게 여러 번 지나가기 → 구간 연결·동선 고정점 찍기 쉬움
- **바닥이 화면에 충분히** 들어오게 (조금 아래를 향하게). 안 그러면 바닥 구멍
- 노출이 흔들리면(밝기 53~187) Brush 는 보정 못 함 → LichtFeld(노출 보정 기본)
