# 인수인계: NoClip 3DGS + IndoorGS (2026-11-27 밤)

다른 Claude 세션이 이어받을 때 이 파일부터 읽는다. 저장소 규칙은 `CLAUDE.md`(커밋 작성자 5undura, 서명 줄 금지, 원본 데이터 삭제·덮어쓰기 금지).

## 1. 목표

- **NoClip : AfterSchool**: 중앙대 학생회관 지하를 3DGS로 스캔한 VR 게임(팀 3명, 하지운 = 3DGS R&D). 지금 목표는 **VR에서 메쉬처럼 보이지 않는 선명한 스플랫**(반투명·비침 줄이기). Unity 쪽 작업은 후순위.
- **IndoorGS**: 위 파이프라인을 일반화한 **실내 공간 원클릭 3DGS 도구**. 영상 → 추출 → 정합(박스·도면 공간 + 브러시 동선으로 위치 보조) → 학습(Brush 기본, LichtFeld 고화질) → 보강 → 한 건물 PLY. 팀용으로 먼저 쓰고, 나중에 공개(오픈소스·포트폴리오).

## 2. 진행 상황

### NoClip 3DGS (브랜치 `main` = `pipeline-upgrade`, PC Claude Code가 작업)
- 복도 1 실험 결론 (`3_train/PROGRESS.md`): 최고 설정은 **LichtFeld v0.5.4.3, 1920px, 300만, SH2, MRNF, `--pose-opt`, 80,000회** → PSNR 26.79, 반투명 49.4% (Brush 기준선 25.75 / 86.9%). 2560px, 400만, global·incremental 재정합, 벽 바탕층 전체는 효과 없음.
- 이전 촬영본은 PC `F:\Sample\4차\up\prev_1\`로 정리, `up\1a, 1b, 1c_*`는 **재촬영본용으로 비워 둠**.
- 재촬영본 처리는 사용자가 PC Claude Code로 따로 진행 예정.
- 확인 안 된 것: 최고 결과(`up\prev_1\1c\L7_80k`, `L7_fill`)를 VR·SuperSplat에서 눈으로 확인, 초록 방(구간 5, 33B117) `F:\Sample\exp5` 실행 결과.

### IndoorGS (브랜치 `indoorgs`, 이 세션에서 작업)
- 계획서 `docs/IndoorGS_PLAN.md`, 약속 `indoorgs/CONTRACT.md`, 사용법 `indoorgs/docs/USAGE.md`, PC 확인 목록 `indoorgs/docs/PC_CHECKLIST.md`.
- 에이전트 7개 작업을 모두 합침, **테스트 159개 통과** (리눅스, 가짜 COLMAP·학습기):
  - 코어: extract, register(`pose_prior_mapper` + 실패 시 `mapper`, 옵션·DB 스키마는 COLMAP 4.2.1 소스로 확인), diagnose(위치 튄 사진·방 밖 카메라 제외), train(어댑터, eval.json·stats.json), CLI(`init/check/status/run/gui`, `--rerun`)
  - 공간: `space/` 박스 편집 모델, DXF·SVG·도면 사진 불러오기, 동선 정리·검사, 동선 → 프레임별 위치(priors.json)
  - 보강·합치기: `plyio`(속성 이름으로 읽기, SH 차수 1~3 회전), fill(바닥·천장 구멍만), merge(도면 좌표 + 문 근처 ICP, 또는 layout_v4)
  - GUI: 로컬 웹 화면 5개(프로젝트·공간·동선·실행·결과), 도면 불러오기·검사, 실행 중지, report.html
  - Windows: `indoorgs/windows/install.bat`, `new_project.bat`, `IndoorGS.bat`, 도구 자동 찾기
- **실제 영상·Windows 도구로는 아직 끝까지 안 돌려 봄.**

## 3. 다음 계획 (순서대로)

1. **PC에서 IndoorGS 설치·점검** — GitHub Desktop으로 `indoorgs` 브랜치 → `indoorgs/windows/install.bat` → `indoorgs/docs/PC_CHECKLIST.md`를 위에서부터. 핵심:
   - `COLMAP.bat pose_prior_mapper -h` (명령 이름, `--overwrite_priors_covariance 0` 형식)
   - Brush 0.3.0 평가 렌더 폴더 이름(`eval_<iter>/`), LichtFeld `metrics.csv` 위치
   - 정션으로 만든 `data_clean/images`를 학습기가 읽는지
   - 4차 복도 1 영상으로 `indoorgs run <proj> --until train` (preset fast)
   - 코드 안 `TODO(PC` 주석 (`git grep -n "TODO(PC"`)
2. **재촬영** (모퉁이·문 앞 2초 멈춤, 천천히, 바닥이 화면에 충분히) + **줄자로 방·복도 치수, 피난 안내도 사진** → IndoorGS 박스 입력·동선 고정점에 바로 씀.
3. **재촬영본으로 IndoorGS 첫 실전**: 공간(박스) + 동선 넣고 `pose_prior_mapper` 정합 vs 기존 정합 비교 (위치 튄 사진 수, PSNR, 구간 이음새).
4. 실제 데이터로 값 맞추기: 도면 사진 인식 기준, 보강 반경 배수(3), 문 근처 ICP 제한(3도 / 0.3m).
5. 다 맞으면 `indoorgs`를 `main`에 합치기 (PC 쪽 작업과 충돌 확인 후).
6. 후순위: Unity(sungjae 병합 커밋, 공간별 컬링 검증, 가구·문 스플랫, HUD), 발표 자료 미정 2건(4번 E 설정, MSAA·SH 반영).

## 4. 아직 정할 것

- IndoorGS 공개 라이선스 (MIT / Apache 2.0). LichtFeld는 GPLv3라 코드에 넣지 않고 별도 exe로만 부름.
- 학생회관 지하 치수 확보 방법 (줄자 실측 / 시설팀 도면 / 피난 안내도 사진).

## 5. 알려진 한계 (IndoorGS)

- 위치 보조는 "튀는 사진 0장, 자동 합치기"용. 2px 수준 미세 자세는 학습기 `--pose-opt`가 맡음.
- Brush 기본값은 반투명이 높아(복도 1: 86.9%) VR에서 비칠 수 있음 → 보고서에 경고, VR용은 LichtFeld 권장.
- 도면 사진 인식은 초안(반자동). 실제 피난 안내도로 시험 안 함.
- Osmo HEVC 영상은 브라우저에 따라 동선 화면에서 재생이 안 될 수 있음 (H.264로 변환 안내).
