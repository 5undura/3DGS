# IndoorGS

실내 공간 영상을 버튼 한 번으로 한 건물 좌표의 3DGS(PLY)로 만드는 도구. 계획: [`docs/IndoorGS_PLAN.md`](../docs/IndoorGS_PLAN.md), 약속(폴더·파일 형식·함수 모양): [`CONTRACT.md`](CONTRACT.md).

**설치·사용법: [`docs/USAGE.md`](docs/USAGE.md)** (Windows, 더블클릭 `windows/install.bat` → `new_project.bat` → `IndoorGS.bat`).
실제 PC 에서 확인할 것: [`docs/PC_CHECKLIST.md`](docs/PC_CHECKLIST.md).

지금 상태 (2026-11-27): **1차 4명(A 코어·CLI, B 공간, C 보강·합치기, D GUI) 작업이 합쳐짐**, 2차(E GUI↔공간 도구, F SH 회전·재실행, G 설치·문서) 진행 중.
- 추출 → 동선 위치 → 정합 → 진단 → 학습(Brush·LichtFeld) → 보강 → 합치기 → 보고서 단계가 모두 코드로 있음
- 로컬 웹 화면 5개(프로젝트·공간·동선·실행·결과), 도구 자동 찾기(`init`·`check`), Windows 더블클릭 bat
- 테스트 137개 통과 (리눅스, 가짜 COLMAP·학습기). **실제 촬영 데이터·Windows 도구로는 아직 끝까지 안 돌려 봄**

개발자용 (터미널):
```
pip install -e ".[all,dev]"                       # 또는 windows/install.bat (.venv)
indoorgs init D:/scan/student_hall --search F:/Sample/tools   # project.yaml, 도구 자동 찾기
indoorgs check D:/scan/student_hall               # 도구·영상 점검, 다른 후보
indoorgs status D:/scan/student_hall              # 구간 × 단계 표
indoorgs run D:/scan/student_hall [--section room_b117] [--until register]
indoorgs gui D:/scan/student_hall --open          # 로컬 웹 화면
PYTHONPATH=src python -m pytest -q
```

## 에이전트 작업 (1차, 병렬 — 끝나서 합쳐짐)

각 에이전트는 자기 파일만 고치고, 다른 모듈과는 CONTRACT.md 4절의 파일로만 주고받는다. 브랜치 `indoorgs`.

**A 코어·CLI** — `stages/extract.py, register.py, diagnose.py, train.py`, `trainers/`, `cli.py`
- 저장소의 `1_extract/pick_best.py`, `2_register/register_all.ps1`, `tools/diagnose.py`, `3_train/*_eval.ps1` 를 파이썬 단계로 옮김
- register: priors.json 이 있으면 COLMAP `pose_prior_mapper` (스키마·옵션은 COLMAP 4.2.1 `-h` 로 확인 후 고정), 실패 시 `mapper`
- 완료 기준: 4차 복도 1 영상으로 `indoorgs run --until train` (preset fast) 이 끝까지 돌고 eval.json·stats.json 이 생김

**B 공간 도구** — `pathing.py`, `stages/prior.py`, `space/` (새로)
- `space/`: 박스 편집 데이터 모델(추가·이동·크기·회전·문 구멍), DXF·SVG 벽 선 → 방 다각형, 도면 사진 → 선 검출 초안(반자동)
- 완료 기준: 박스 3개 + 문 2개를 코드로 만들어 plan.json 저장·읽기 왕복, DXF 예제 하나에서 방 다각형 추출

**C 보강·합치기** — `stages/fill.py, merge.py`
- `tools/fill/fill_section.py`, `merge_holes.py` 를 함수로, 바닥·천장 높이는 plan.json 이 있으면 그 값
- merge: 공간 정합 구간은 변환 없이 합치고 문 근처 미세 정렬, 없으면 layout.json
- 완료 기준: 4차 결과 PLY 두 구간을 `tools/layout_v4.json` 으로 합친 building.ply 가 기존 `place.py` 결과와 같음

**D GUI** — `gui/` (새로), `stages/report.py`
- 로컬 웹(FastAPI) 화면 4개: 프로젝트 / 공간(박스 편집 + 3D 미리보기, 브러시 동선, 영상 보며 고정점 찍기) / 실행 / 결과
- 처음엔 `examples/demo` 를 가짜 데이터로 화면부터, 그다음 `stages.base.status`·`run_all` 연결
- 완료 기준: demo 프로젝트에서 박스를 옮기고 동선을 그려 저장하면 plan.json·path.json 이 계약 형식으로 바뀜
- 2차(E): 공간 화면 "도면 불러오기"(DXF·SVG·사진 → 초안 미리보기 → 바꾸기/추가하기, 원본은 `floorplan/source/`), 오른쪽 검사 목록(`/api/plan/validate`), 방을 옮기면 그 방 벽의 문이 따라감, 배경 정렬은 `source.image_align`
