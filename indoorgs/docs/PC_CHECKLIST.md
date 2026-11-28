# Windows PC 에서 확인할 것

리눅스 테스트로는 확인 못 한 것들. 위에서부터 순서대로 (앞 항목이 막히면 뒤도 못 함).
명령은 PowerShell 기준, `$C = "F:\Sample\tools\colmap\COLMAP.bat"`, `$IG = "<저장소>\.venv\Scripts\indoorgs.exe"`.
확인한 결과는 해당 코드 주석(`TODO(PC…`)을 지우고 버전·날짜를 적는다. 새 항목 찾기: `git grep -n "TODO(PC"`.

## 0. 설치·실행 파일 (G)
- [ ] `windows\install.bat` 더블클릭 → `.venv` 생김, 끝에 "Done". 두 번째 실행은 "already exists - skipped"
- [ ] `windows\new_project.bat` → `test1` 입력 → `project.yaml` 의 tools 3개가 `F:/Sample/tools/...` 로 채워짐. LichtFeld 는 `build\` 아래 실제 exe 경로인지
- [ ] `& $IG check <proj>` → "자동으로 찾은 다른 후보" 줄이 실제 다른 사본만 보여 줌 (예: 예전 COLMAP)
- [ ] `windows\IndoorGS.bat` 에 프로젝트 폴더 끌어다 놓기 → 브라우저가 열림. 한글 경로 폴더도 열리는지
- [ ] bat 실행 중 글자 깨짐 없음 (bat 은 ASCII, 파이썬 출력은 한국어)

## 1. COLMAP (A)
- [ ] `& $C help | Select -First 1` → `COLMAP 4.2.1 (Commit ...)` 형식. 다르면 `cli.py colmap_version` 정규식 수정
  (코드: `cli.py` `colmap_version`)
- [ ] `& $C pose_prior_mapper -h` → 명령이 있고 `--overwrite_priors_covariance` 옵션이 있음. 값을 `0`/`1` 로 받는지 (`true`/`false` 만 받으면 `register.py PRIOR_MAPPER_EXTRA` 수정)
  (코드: `stages/register.py` `CMD_PRIOR_MAPPER`, `PRIOR_MAPPER_EXTRA`)
- [ ] `& $C sequential_matcher -h | Select-String loop_detection` → `--SequentialMatching.loop_detection` 있음. 처음 실행 때 어휘 트리를 자동으로 받는지, **오프라인에서는 실패하는지** (실패하면 `register: {loop_detection: false}` 안내가 맞는지)
  (코드: `stages/register.py` `OPT_SEQ_LOOP`)
- [ ] (anchored 실험 쓸 때만) `& $C matches_importer -h` → `--match_list_path`, `--match_type`; `& $C point_triangulator -h` → `--input_path --output_path --clear_points`;
  `& $C bundle_adjuster -h` → `--BundleAdjustmentCeres.max_num_iterations`, `--BundleAdjustmentCeres.use_gpu`; `& $C image_registrator -h` → `--input_path --output_path` (`--image_path` 없음);
  `& $C pose_prior_mapper -h | Select-String ba_use_gpu` → `--Mapper.ba_use_gpu`. 점 없는 초기 모델(frames.bin·rigs.bin 없이)을 point_triangulator 가 읽는지
  (코드: `stages/register.py` `CMD_MATCHES_IMPORTER` ~ `OPT_MAPPER_GPU`, `experiments/anchored/RESULTS.md`)
- [ ] 공간+동선 있는 구간의 pose_prior_mapper 결과가 **도면 좌표**로 나오는지: `sections\<s>\colmap\sparse\0` 의 카메라 위치를 report.html 도면 위 점으로 보고 동선과 겹치는지 (기대: 어긋남 1m 이내)

## 2. 진단·학습 입력 (A, C)
- [ ] 정션으로 만든 `data_clean\images` 를 Brush·LichtFeld 가 읽는지: `dir sections\<s>\data_clean` 에 `<JUNCTION> images` 표시, 학습 로그의 사진 수 = 정합 장수 − 뺀 장수
- [ ] `data_clean` 모델이 PINHOLE 이고 사진이 `data_clean\images` 에 있음: `& $C model_analyzer --path sections\<s>\data_clean\sparse\0` → cameras 모델 PINHOLE

## 3. 학습기 (A, C)
- [ ] Brush 0.3.0 `--eval-save-to-disk` 렌더 폴더·이름 = `eval_<iter>\<사진이름>.png` (기대: eval.json 의 psnr 이 null 아님)
  (코드: `trainers/brush.py` `evaluate`)
- [ ] LichtFeld `metrics.csv` 위치: 출력 폴더 아래 어딘가 (`Get-ChildItem -Recurse -Filter metrics.csv <run>`), 마지막 줄 psnr 이 eval.json 과 같음
- [ ] Brush·LichtFeld 출력 PLY 가 COLMAP 좌표 그대로인지 (특히 LichtFeld `--pose-opt`): SuperSplat 에서 PLY 와 `sparse\0` 점(또는 report 카메라 위치)이 겹침
- [ ] **완료 기준**: 4차 복도 1 영상(`up` 에 복사해서)으로 `& $IG run <proj> --until train` (preset `fast`) 이 끝까지 돌고 `eval.json`·`stats.json` 생김. 10분 넘으면 로그 파일로 백그라운드 실행

## 4. 보강·합치기 (C)
- [ ] layout_v4 로 합친 `merged\building.ply` 가 팀원 `place.py` 결과와 같은지 (SH 회전·unity_fix): 두 PLY 의 점 수가 같고 같은 순서 점의 위치 차이 1mm 이내, SH 계수(f_rest)도 같음, Unity 에 넣었을 때 같은 자리. 다르면 unity_fix(merge 는 적용 안 하고 layout.json notes 에만 적음) 차이인지 먼저 볼 것
- [ ] fill 결과(`fill\holes_backdrop.ply`, 반경 배수 3)를 VR 에서 확인: 바닥·천장 구멍이 메워지고 벽이 뭉개지지 않음
- [ ] 문 근처 ICP: `merged\layout.json` notes 의 수용 비율과 보정량이 제한(회전 3도·이동 0.3m) 안. 제한에 자주 걸리면 값 조정
- [ ] 300만 × 6구간 합칠 때 메모리: 작업 관리자에서 최대 메모리 (기대: PC 메모리 안에서 끝남. 넘치면 merge 를 나눠 쓰는 방식이 필요 → C 에게)

## 5. GUI (D)
- [ ] Osmo HEVC 10bit 영상이 동선 화면에서 재생되는지 (Chrome·Edge). 안 되면 USAGE.md 문제 해결의 H.264 미리보기 변환 안내(ffmpeg: `F:\Sample\tools\ffmpeg_tool`)가 통하는지
