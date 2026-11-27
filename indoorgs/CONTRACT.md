# IndoorGS 계약 (설계 0단계)

여러 에이전트가 동시에 작업하기 전에 고정하는 약속이다. 여기 적힌 **폴더 구조, 파일 형식, 함수 모양**만 지키면 모듈끼리 서로의 코드를 몰라도 된다. 바꿔야 하면 이 문서를 먼저 고치고 모든 작업자에게 알린다. 계획 전체는 [`docs/IndoorGS_PLAN.md`](../docs/IndoorGS_PLAN.md).

## 1. 코드 위치

```
indoorgs/                      이 폴더만 IndoorGS. 저장소의 기존 스크립트(1_extract, 2_register, 3_train, tools)는 건드리지 않음
  pyproject.toml
  CONTRACT.md                  이 문서
  src/indoorgs/
    config.py                  project.yaml 읽기·검사          (공통)
    project.py                 프로젝트·구간 경로              (공통)
    schemas.py                 plan.json · path.json · priors.json · diagnose.json 형식  (공통)
    pathing.py                 동선 → 프레임별 위치 (구현됨)    (B)
    stages/base.py             Stage 규격, 실행 순서            (공통)
    stages/extract.py          (A)    stages/register.py    (A)    stages/diagnose.py   (A)
    stages/prior.py            (B)    stages/train.py       (A)
    stages/fill.py             (C)    stages/merge.py       (C)    stages/report.py     (D)
    trainers/base.py, brush.py, lichtfeld.py                 (A)
    space/                     박스 편집·도면 불러오기          (B, 새로 만듦)
    gui/                       로컬 웹 UI                      (D, 새로 만듦)
    cli.py                     indoorgs init / status / run     (A)
  tests/
  examples/demo/               형식 예시 (영상 없음)
```

괄호는 담당: **A 코어·CLI, B 공간 도구, C 보강·합치기, D GUI**. 공통 파일은 바꾸기 전에 이 문서부터 고친다.

## 2. 프로젝트 폴더 (사용자 데이터)

```
<project>/
  project.yaml
  floorplan/plan.json          공간 (방·문). 없으면 공간 기능 꺼짐
  floorplan/plan.png           도면 이미지 (선택)
  sections/<name>/
    path.json                  동선 (선택)
    frames/  frames/done.json  extract
    colmap/priors.json         prior
    colmap/db.db  colmap/sparse/<n>/   register
    data/images  data/sparse/0         register (왜곡 보정 사진, 학습 입력)
    diagnose.json  data_clean/         diagnose
    train/<trainer>_<preset>/splat.ply  eval.json  stats.json   train
    fill/holes_backdrop.ply            fill
  merged/building.ply  merged/layout.json   merge
  report.html                               report
  logs/<section>_<stage>.log
```

- 경로는 영문 권장. 한글 경로는 OpenCV `imwrite`/`imread`, COLMAP에서 조용히 실패한 적이 있다. 사진 읽기·쓰기는 `cv2.imdecode`/`imencode`로.
- 원본(영상, 이전 단계 결과)은 지우거나 덮어쓰지 않는다. 다시 돌리면 `train/brush_default_2`처럼 새 폴더.
- 실패한 단계의 출력 폴더는 `<이름>_failed_<시각>`으로 옆으로 치운다.

## 3. 좌표

- 단위 미터. 공간이 있으면 **도면 좌표**: x 오른쪽, y 위쪽(도면을 위에서 봄), z 위, 바닥 z = 0.
- COLMAP·학습기 좌표는 공간이 없으면 각자 좌표. merge가 구간별 4x4 변환을 `merged/layout.json`에 적는다.

## 4. 파일 형식

모든 JSON은 UTF-8, 최상위에 `"version": 1`.

### project.yaml

[`examples/demo/project.yaml`](examples/demo/project.yaml) 참고. 필드와 기본값은 `config.py`의 데이터클래스가 기준이다.

### frames/done.json (extract가 씀)

```json
{"version": 1, "video": "D:/capture/5.MP4", "fps": 23.976, "total_frames": 6488,
 "count": 1540, "args": {"min_diff": 4, "sharp_ratio": 0.8}}
```

프레임 파일명은 `f_<순번 5자리>_<원본 프레임 번호 6자리>.jpg`. 시간(초) = 원본 프레임 번호 / fps.

### floorplan/plan.json (B가 씀)

```json
{"version": 1, "units": "m",
 "rooms": [
   {"id": "corridor1", "name": "복도 1", "box": {"center": [0, 0], "size": [30, 2.4], "yaw_deg": 0}, "floor_z": 0, "ceiling_z": 2.7},
   {"id": "b117", "floor": [[-1.2, 2], [-1.2, 9], [-6, 9], [-6, 2]], "floor_z": 0, "ceiling_z": 2.6}],
 "doors": [{"id": "b117_s", "rooms": ["corridor1", "b117"], "center": [-1.2, 3.97], "width": 1.0, "height": 2.1}],
 "source": {"kind": "boxes"}}
```

- 방은 `box`(중심·크기·회전) 또는 `floor`(바닥 다각형, 반시계) 중 하나. 읽을 때 `box`는 `floor`로 바뀐다.
- `source.kind`: `boxes` | `dxf` | `image`. 이미지면 `image`, `px_per_m`, 원근 보정 네 점을 함께 적는다.

### sections/<name>/path.json (B가 씀)

```json
{"version": 1, "camera_height": 1.4, "sigma_xy": 0.5, "sigma_z": 0.3,
 "stroke": [[0, 0], [10, 0], [10, 5]],
 "anchors": [{"t": 0.0, "point": [0, 0]}, {"t": 42.5, "point": [10, 0]}, {"t": 60.0, "s": 15.0}]}
```

- `stroke`: 브러시로 그린 선(걸은 순서). 없으면 고정점들을 직선으로 이은 것.
- `anchors`: 고정점. `point`(도면 좌표)는 선 위 가장 가까운 곳으로 붙고, `s`는 선 시작부터의 거리(m). `t`는 영상 시간(초).
- 고정점 사이는 선을 따라 일정한 속도. 첫 고정점 전·마지막 뒤는 그 자리에 멈춘 것으로 본다. 멈춤은 같은 위치에 고정점 두 개.

### colmap/priors.json (prior가 씀, register가 읽음)

```json
{"version": 1, "coordinate_system": "CARTESIAN",
 "frames": [{"image": "f_00001_000000.jpg", "t": 0.0, "xyz": [0, 0, 1.4], "sigma": [0.5, 0.5, 0.3]}]}
```

register는 이걸 COLMAP DB `pose_priors` 표에 넣고 `pose_prior_mapper`를 돌린다. 표 스키마·옵션 이름은 설치된 COLMAP에서 `-h`로 확인해 `stages/register.py` 맨 위 주석에 적는다.

### diagnose.json (diagnose가 씀)

```json
{"version": 1, "registered": 1072, "total": 1084, "reproj_error_px": 0.96,
 "teleports": [{"from": "f_00779_...", "to": "f_00795_...", "speed_x_median": 14.2}],
 "outside_space": ["f_00412_..."], "far_from_prior": [], "excluded": ["..."]}
```

`excluded`에 든 사진은 `data_clean/`에서 빠진다 (원본 `data/`는 그대로).

### train/<trainer>_<preset>/eval.json, stats.json (train이 씀)

```json
{"version": 1, "psnr": 26.79, "ssim": 0.885, "n_eval": 133, "eval_every": 8}
{"version": 1, "count": 3000000, "translucent_ratio": 0.494, "needle_ratio": 0.013}
```

`translucent_ratio` = 불투명도 0.5 미만 비율, `needle_ratio` = 가장 긴 축 / 가장 짧은 축 > 10 비율 (`tools/ply_stats.py`와 같은 정의).

## 5. 함수 모양

```python
class Stage:
    name: str                      # "extract" ...
    scope: "section" | "project"   # merge, report 는 project
    uses_gpu: bool                 # True 면 실행기가 한 번에 하나씩만 돌림
    def outputs(self, ctx) -> list[Path]   # 끝났는지 판단할 파일
    def is_done(self, ctx) -> bool         # 기본: outputs 가 모두 있으면 True
    def run(self, ctx) -> StageResult      # 실패는 예외 대신 StageResult(ok=False, message=...)

class TrainerAdapter:
    name: str
    def detect(self, exe) -> TrainerInfo           # 버전, --help 에 필요한 옵션이 있는지
    def presets(self) -> dict[str, Preset]
    def command(self, exe, data_dir, out_dir, preset, gpu) -> list[str]
    def find_result(self, out_dir) -> Path | None  # 최종 PLY
```

`ctx`는 `StageContext(project, section, cfg, log)`. 단계 순서: extract → prior → register → diagnose → train → fill → (모든 구간 후) merge → report. `prior`는 `plan.json`과 `path.json`이 둘 다 있을 때만 돈다.

## 6. 작업 규칙

- 외부 도구 옵션 이름은 추측하지 않는다. `--help`/`-h`로 확인하고, 확인한 버전을 코드 주석에 적는다.
- 개발 중 GPU 검증은 짧게(학습 5,000회). 긴 학습은 A가 한 번에 하나씩.
- 커밋 작성자 5undura, 커밋 메시지에 서명 줄 없음. IndoorGS 작업은 `indoorgs` 브랜치.
- 새 의존성은 `pyproject.toml`에 추가하고 이유를 커밋 메시지에 적는다. 시스템 설치(CUDA, VS 등)는 사용자에게 먼저 묻는다.
