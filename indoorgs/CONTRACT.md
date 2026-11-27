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
    walls/planes.json  data_seed/      walls (도면 좌표일 때만: 면 다시 맞추기 + 초기 점 심기)
    train/<trainer>_<preset>/splat.ply  eval.json  stats.json   train
    train/<trainer>_<preset>/splat_walls.ply  walls.json      snap (벽에 붙이기, 원본 splat.ply 는 그대로)
    train/<trainer>_<preset>/splat_clean.ply  clean.json  removed_clean.ply   cleanup (학습 후 정리, 모든 구간)
    train/<trainer>_<preset>/splat_repaired.ply  repair.json  repair (면 메우기)
    fill/holes_backdrop.ply            fill
  merged/building.ply  merged/layout.json   merge
  report.html                               report
  logs/<section>_<stage>.log
```

- 경로는 영문 권장. 한글 경로는 OpenCV `imwrite`/`imread`, COLMAP에서 조용히 실패한 적이 있다. 사진 읽기·쓰기는 `cv2.imdecode`/`imencode`로.
- 원본(영상, 이전 단계 결과)은 지우거나 덮어쓰지 않는다. 다시 돌리면 `train/brush_default_2`처럼 새 폴더.
  - `indoorgs run <proj> --rerun <단계>`(`run_all(rerun=)`)로만 다시 돈다. train 은 최신 폴더가 끝났으면 `_2`, `_3`… 새 폴더,
    다른 단계는 이전 결과(`Stage.prev_targets`)를 `<이름>_prev_<시각>`으로 옮긴 뒤 다시.
  - fill·merge·report 는 `SectionPaths.latest_train_run(trainer, preset)` = splat.ply 가 있는 가장 큰 번호 폴더를 쓴다.
  - 취소: `run_all(cancel=threading.Event)` 단계 사이에서 멈추고 `runner.run` 중인 외부 프로그램을 끝냄 (GUI `POST /api/run/cancel`).
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

- 방은 `box`(중심·크기·회전) 또는 `floor`(바닥 다각형, 반시계) 중 하나. 읽을 때 `box`로 `floor`를 계산하고, `Room.box`에 박스 정보도 남겨 다시 쓸 때 `box`로 저장한다.
- `source.kind`: `boxes` | `dxf` | `svg` | `image`. 이미지면 `px_per_m`, 원근 보정 네 점(`corners`)·`size_m`을 함께 적는다.
- `source.image_align` (선택): 도면 배경(`plan.png`) 정렬 `{"px_per_m": 50, "origin_px": [x, y], "flip_y": true}`. `origin_px`는 도면 (0, 0)이 놓이는 이미지 픽셀, `flip_y`면 이미지 y 아래 = 도면 y 위. GUI가 읽고 씀. 불러온 원본 도면은 `floorplan/source/`에 보관.
- 편집·불러오기 API는 `indoorgs.space` (`SpaceModel`, `import_dxf/svg/image`).

### sections/<name>/path.json (B가 씀)

```json
{"version": 1, "camera_height": 1.4, "sigma_xy": 0.5, "sigma_z": 0.3,
 "stroke": [[0, 0], [10, 0], [10, 5]],
 "anchors": [{"t": 0.0, "point": [0, 0]}, {"t": 42.5, "point": [10, 0]}, {"t": 60.0, "s": 15.0}]}
```

- `stroke`: 브러시로 그린 선(걸은 순서). 없으면 고정점들을 직선으로 이은 것.
- `anchors`: 고정점. `point`(도면 좌표)는 선 위 가장 가까운 곳으로 붙고, `s`는 선 시작부터의 거리(m). `t`는 영상 시간(초).
- 고정점 사이는 선을 따라 일정한 속도. 첫 고정점 전·마지막 뒤는 그 자리에 멈춘 것으로 본다. 멈춤은 같은 위치에 고정점 두 개.

### colmap/priors_warnings.json (prior가 씀, GUI·report가 읽음)

```json
{"version": 1, "warnings": ["..."], "outside_frames": ["f_..."], "outside_segments": [{"s0": 3.1, "s1": 4.0, "start": [x, y], "end": [x, y], "length": 0.9}], "video_duration": 120.0}
```

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

### walls/planes.json (walls가 씀), data_seed/

도면 좌표(`register.json` mapper = `pose_prior_mapper`)이고 `project.yaml` `walls.enabled`일 때만.

```json
{"version": 1, "rooms": ["corridor1"], "source_model": "data_clean",
 "planes": [{"id": "corridor1:w1", "kind": "wall", "room": "corridor1", "n": [-1, 0, 0], "d": 1.17,
             "origin": [-1.17, -1, 0], "U": [0, 1, 0], "V": [0, 0, 1], "size": [30, 2.7], "holes": [[4.47, 5.47, 0, 2.1]]}],
 "plan_planes": [...], "fit": [{"id": "corridor1:w1", "source": "fit", "shift_m": 0.04, "tilt_deg": 0.3, "n_points": 812, "coverage": 0.6}],
 "points_flatness_before": {...}, "points_flatness_after": {...}, "seed": {"n_added": 182000}}
```

- 면: `n·x = d`, n 은 방 안쪽. 벽은 방 다각형의 변 하나, 바닥·천장은 방 하나 (`walls/planes.py`).
- `planes`는 정합 점으로 다시 맞춘 면(점이 모자라면 도면 그대로, `source: "plan"`), `plan_planes`는 도면 그대로.
- `fit[].source`: `"fit"`(정합 점으로 맞춤; `relaxed`·`offset_only`·`n_front_excluded` 는 느슨한 기준·기울기 0·튀어나온 물체로 뺀 점 수),
  `"infer"`(`walls.trust_plan_dims` 켰을 때만: 맞춘 평행 면 `inferred_from` 이 옮긴 만큼 옮김), `"plan"`(도면 그대로, `reason`).
- `data_seed/`: `data_clean/`과 같은 사진(링크) + `sparse/0`에 면 위 초기 점(색은 사진에서, 트랙 없음)을 더한 모델. 초기 점은 맞춘 면(`fit[].source` 가 `"fit"` 또는 `"infer"`)에만.
  train은 `data_seed` > `data_clean` > `data` 순으로 읽되, `data_seed`는 walls 가 이 구간에 적용(`walls.enabled`, 도면 좌표)되고 `walls.seed`가 켜져 있고
  `source_identity`(`{"n_images", "images_bin_sha1"}`, 만든 원본 모델)가 지금 `data_clean`(없으면 `data`) 모델과 같을 때만 쓴다.
- snap·repair 는 맞춘 면(`fit`/`infer`)만 쓴다 (도면 그대로인 면은 실제 위치를 모름). 둘 다 기본 꺼짐 (`walls.snap`, `walls.repair`; repair 는 snap 필요).
  repair 는 카메라가 있으면 학습 카메라 광선 덮임(면 뒤 0.5 m 까지)으로 구멍을 다시 확인한다 (`repair.json` `faces[].covered_by_rays_m2`). `--rerun` 이 앞 단계부터면 walls·snap·repair 는 적용 안 돼도 이전 결과를 `_prev_`로 옮긴다.
- "도면 좌표인가"는 모든 단계에서 `stages.diagnose.model_in_plan_coords` (= `register.json` mapper) 하나로 판정한다.

### train/<run>/splat_walls.ply, walls.json (snap이 씀)

`walls.snap` 이 켜졌을 때만 (기본 끔, 11-28). (`walls.remove_behind` 면) 벽 뒤(문 구멍 아닌 곳)·바닥 아래·천장 위 부유물을 지우고, 면에서 `walls.band` 안이고
그 자리 벽 색(칸별 넓이 가중 중앙값)과 `walls.color_tol` 안인 스플랫을 면 위로 옮긴 (`walls.flatten` 이면 법선 방향으로 납작하게도 한) PLY. 속성 순서는 원본 그대로. `walls.json`에 붙인 수·지운 수·벽 평탄도(불투명 스플랫의 면까지 거리 RMS) 전후, 쓴 면(`planes_used`), 경고(`warnings`).
불투명 스플랫 절반 넘게 도면 방 밖이면 PLY 없이 `walls.json`에 `"skipped": "<이유>"`만 쓴다.
fill·merge는 `splat_repaired.ply` > `splat_clean.ply` > `splat_walls.ply` > `splat.ply` 중 설정이 켜져 있고 바로 앞 입력보다 새것을 쓴다
(walls ≥ splat.ply, clean ≥ 그 입력(유효한 walls, 없으면 splat.ply), repaired ≥ 그 입력(유효한 clean, 없으면 walls)). `clean.json`·`repair.json`의
`"input"`(입력 파일 이름)이 지금 고른 입력과 다르면(예: snap 이나 cleanup 을 끔) 그 결과도 안 쓴다 (`stages/snap.best_ply`).

### train/<run>/splat_clean.ply, clean.json, removed_clean.ply (cleanup이 씀)

`walls/cleanup.cleanup_file`: 부유물(floaters)·아주 큰 것(huge)·바늘 쪼개기(needles)·SH 튐(sh), `project.yaml` `cleanup.steps` 순서 (기본 `[floaters]`, 11-28). 도면 좌표가 아니어도 돈다.
도면 좌표면 planes.json 면(보호용이라 맞추지 못한 면도 도면 위치로)과 plan 을 넘긴다. `clean.json`: `input`, `plan_coords`, 단계별 정보, `quality_before/after`(count 포함 — 바늘 쪼개기로 수가 늘어남).
`removed_clean.ply`는 지운 스플랫만 (사람이 확인용).

### train/<trainer>_<preset>/eval.json, stats.json (train이 씀)

```json
{"version": 1, "psnr": 26.79, "ssim": 0.885, "n_eval": 133, "eval_every": 8}
{"version": 1, "count": 3000000, "translucent_ratio": 0.494, "needle_ratio": 0.013}
```

`translucent_ratio` = 불투명도 0.5 미만 비율, `needle_ratio` = 가장 긴 축 / **중간 축** > 10 비율 (`tools/ply_stats.py` 코드와 같은 정의. 복도 1 기록 16.3% 등도 이 기준).

### colmap/register.json (register가 씀)

어떤 정합 방식을 썼는지 기록: `pose_prior_mapper`(도면 좌표) / `mapper` / `global_mapper`, 전환했다면 그 이유. diagnose는 `pose_prior_mapper`일 때만 공간 검사(방 밖, 위치 제약에서 1m)를 한다. 다른 좌표에서는 의미가 없기 때문이다.

### fill/fill.json (fill이 씀)

바닥·천장 면의 출처(`plan.json` 또는 점에서 추정), 추가한 바탕층 수, 임계값(반경 배수 3, 덮임 기준 1.0).

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

`ctx`는 `StageContext(project, section, cfg, log)`. 단계 순서: extract → prior → register → diagnose → walls → train → snap → cleanup → repair → fill → (모든 구간 후) merge → report. `walls`·`snap`·`repair`는 도면 좌표 구간에서만, `cleanup`은 모든 구간에서 돈다 (`cleanup.enabled`). `prior`는 `plan.json`과 `path.json`이 둘 다 있을 때만 돈다.

## 6. 작업 규칙

- 외부 도구 옵션 이름은 추측하지 않는다. `--help`/`-h`로 확인하고, 확인한 버전을 코드 주석에 적는다.
- 개발 중 GPU 검증은 짧게(학습 5,000회). 긴 학습은 A가 한 번에 하나씩.
- 커밋 작성자 5undura, 커밋 메시지에 서명 줄 없음. IndoorGS 작업은 `indoorgs` 브랜치.
- 새 의존성은 `pyproject.toml`에 추가하고 이유를 커밋 메시지에 적는다. 시스템 설치(CUDA, VS 등)는 사용자에게 먼저 묻는다.
