"""GUI 서버 (FastAPI). plan.json · path.json 을 직접 읽고 쓰며 schemas 로 검사한다.

API (모두 /api 아래)
  GET  project                      설정 요약, 구간 목록
  GET  plan / PUT plan              floorplan/plan.json (잘못되면 400 + 한국어 메시지)
  GET  plan/image                   floorplan/plan.png
  POST plan/import                  도면 업로드(.dxf .svg 사진) -> 초안 (plan.json 은 안 씀)
  GET  plan/validate / POST         저장된(또는 보낸) plan 검사 -> issues
  GET  sections/{name}/path / PUT   sections/<name>/path.json
  GET  sections/{name}/video        project.yaml 의 영상 (Range 지원)
  GET  status                       구간 × 단계 (done/todo/skip/running/fail)
  POST run  {section?, until?}      백그라운드로 run_all (동시에 하나만)
  POST run/cancel                   실행 취소 (단계 사이 + 실행 중인 외부 프로그램 종료)
  GET  run/state
  GET  logs                         logs/*.log 목록
  GET  logs/{file}?n=200            꼬리 n줄
  GET  report?rebuild=1             report.html
"""
from __future__ import annotations

import datetime as dt
import mimetypes
import re
import threading
from collections import deque
from pathlib import Path

import json

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import schemas
from ..project import Project
from ..stages.base import StageContext, all_stages, run_all, status
from . import plan_import

STATIC = Path(__file__).parent / "static"
LOG_NAME = re.compile(r"^[A-Za-z0-9_\-]+\.log$")
CHUNK = 1 << 20


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _tail(path: Path, n: int) -> list[str]:
    # 큰 로그도 끝부분만 읽음
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 256 * 1024))
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-n:]


class Runner:
    """run_all 을 백그라운드 스레드 하나로. 동시에 하나만."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.cancel_ev = threading.Event()
        self.lines: deque[str] = deque(maxlen=500)
        self.state: dict = {"running": False}

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, section: str | None, until: str | None) -> dict:
        with self.lock:
            if self.busy():
                raise HTTPException(409, "이미 실행 중")
            project = Project(self.root)        # 설정을 새로 읽음
            if section:
                project.section(section)        # 없으면 KeyError
            names = [s.name for s in all_stages()]
            if until and until not in names:
                raise HTTPException(400, f"모르는 단계 {until} (가능: {names})")
            self.lines.clear()
            self.state = {"running": True, "section": section, "until": until,
                          "started": dt.datetime.now().isoformat(timespec="seconds"),
                          "finished": None, "ok": None, "error": None}
            self.cancel_ev = threading.Event()
            self.thread = threading.Thread(target=self._work, args=(project, section, until), daemon=True)
            self.thread.start()
            return dict(self.state)

    def _work(self, project, section, until):
        try:
            ok = run_all(project, only_section=section, until=until, echo=self.lines.append, cancel=self.cancel_ev)
            self.state["ok"] = ok
        except Exception as e:                  # 단계 밖 오류도 화면에 보이게
            self.state["ok"] = False
            self.state["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.state["running"] = False
            self.state["finished"] = dt.datetime.now().isoformat(timespec="seconds")

    def snapshot(self) -> dict:
        return dict(self.state) | {"lines": list(self.lines)}

    def cancel(self) -> dict:
        """실행 취소 요청 (F). run_all 이 단계 사이에서 멈추고 실행 중인 외부 프로그램도 끝냄."""
        if not self.busy():
            raise HTTPException(409, "실행 중이 아님")
        self.cancel_ev.set()
        self.state["cancel_requested"] = True
        return dict(self.state)


def create_app(project_root: str | Path) -> FastAPI:
    root = Path(project_root).resolve()
    Project(root)                                # 설정 오류는 시작할 때 바로
    app = FastAPI(title="IndoorGS")
    runner = Runner(root)
    app.state.runner = runner

    def proj() -> Project:
        try:
            return Project(root)
        except Exception as e:
            raise HTTPException(500, f"project.yaml 읽기 실패: {e}")

    def section_paths(p: Project, name: str):
        try:
            return p.section(name)
        except KeyError:
            raise HTTPException(404, f"구간 없음: {name}")

    def put_json(path: Path, data, parse, name: str):
        if not isinstance(data, dict):
            raise HTTPException(400, f"{name}: JSON 객체여야 함")
        data = dict(data)
        data.setdefault("version", schemas.VERSION)
        try:
            parse(data)
        except schemas.SchemaError as e:
            raise HTTPException(400, str(e))
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"{name}: 형식 오류 ({type(e).__name__}: {e})")
        schemas.write_json(path, data)
        return {"ok": True, "path": str(path)}

    @app.exception_handler(KeyError)
    async def _key_error(request: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    # ---------- 프로젝트 ----------
    @app.get("/api/project")
    def get_project():
        p = proj()
        c = p.cfg
        secs = []
        for s, sp in p.sections():
            secs.append({"name": s.name, "video": s.video, "video_exists": Path(s.video).exists(),
                         "has_path": sp.path_json.exists(), "has_frames": sp.frames_done.exists()})
        return {"root": str(p.root), "name": p.root.name, "has_space": p.has_space,
                "has_plan_image": p.plan_image.exists(), "warnings": p.warnings(),
                "gpu_profile": c.gpu_profile, "tools": vars(c.tools),
                "train": vars(c.train), "register": vars(c.register),
                "extract": vars(c.extract), "fill": vars(c.fill),
                "stages": [s.name for s in all_stages()], "sections": secs,
                "has_report": p.report.exists()}

    # ---------- 공간 ----------
    @app.get("/api/plan")
    def get_plan():
        p = proj()
        if not p.plan_json.exists():
            raise HTTPException(404, "plan.json 없음 (공간을 새로 만들 수 있음)")
        return schemas.read_json(p.plan_json)

    @app.put("/api/plan")
    def put_plan(data: dict = Body(...)):
        out = put_json(proj().plan_json, data, schemas.parse_plan, "plan.json")
        out["issues"] = plan_import.validate_plan(data)       # 저장은 막지 않고 알려만 줌
        return out

    @app.get("/api/plan/validate")
    def validate_saved():
        p = proj()
        if not p.plan_json.exists():
            raise HTTPException(404, "plan.json 없음")
        return {"issues": plan_import.validate_plan(schemas.read_json(p.plan_json))}

    @app.post("/api/plan/validate")
    def validate_body(data: dict = Body(...)):
        return {"issues": plan_import.validate_plan(data)}

    @app.post("/api/plan/import")
    async def import_plan(file: UploadFile = File(...), options: str = Form("{}")):
        """options(JSON): px_per_m, scale_m, scale_points, corners, size_m, save_png, recenter, min_area, snap_tol"""
        try:
            opt = json.loads(options or "{}")
            if not isinstance(opt, dict):
                raise ValueError
        except ValueError:
            raise HTTPException(400, "options 는 JSON 객체여야 함")
        data = await file.read()
        try:
            return plan_import.run_import(proj().plan_json.parent, file.filename or "plan", data, opt)
        except plan_import.ImportError_ as e:
            raise HTTPException(400, str(e))
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"옵션 오류: {e}")

    @app.get("/api/plan/image")
    def plan_image():
        p = proj()
        if not p.plan_image.exists():
            raise HTTPException(404, "plan.png 없음")
        return FileResponse(p.plan_image, media_type="image/png")

    # ---------- 동선 ----------
    @app.get("/api/sections/{name}/path")
    def get_path(name: str):
        sp = section_paths(proj(), name)
        if not sp.path_json.exists():
            raise HTTPException(404, "path.json 없음")
        return schemas.read_json(sp.path_json)

    @app.put("/api/sections/{name}/path")
    def put_path(name: str, data: dict = Body(...)):
        sp = section_paths(proj(), name)
        return put_json(sp.path_json, data, schemas.parse_path, "path.json")

    @app.get("/api/sections/{name}/video")
    def get_video(name: str, request: Request):
        p = proj()
        section_paths(p, name)
        sec = next(s for s in p.cfg.sections if s.name == name)
        video = Path(sec.video)
        if not video.is_absolute():
            video = p.root / video
        # project.yaml 에 적힌 영상만 보냄 (요청 경로를 쓰지 않음)
        if not video.is_file():
            raise HTTPException(404, f"영상 없음: {sec.video}")
        return _range_response(video, request.headers.get("range"))

    # ---------- 실행 ----------
    @app.get("/api/status")
    def get_status():
        p = proj()
        rows = []
        running = runner.busy()
        for sec, stage, state in status(p):
            log = p.logs / f"{'project' if sec == '*' else sec}_{stage}.log"
            if state == "todo" and log.exists():
                last = (_tail(log, 1) or [""])[0]
                if ": 실패" in last:
                    state = "fail"
                elif running and last.endswith(": 시작"):
                    state = "running"
            rows.append({"section": sec, "stage": stage, "state": state,
                         "log": log.name if log.exists() else None})
        return {"rows": rows, "stages": [s.name for s in all_stages()], "running": running}

    @app.post("/api/run")
    def post_run(body: dict | None = Body(default=None)):
        body = body or {}
        try:
            return runner.start(body.get("section") or None, body.get("until") or None)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))

    @app.post("/api/run/cancel")
    def post_run_cancel():
        return runner.cancel()

    @app.get("/api/run/state")
    def run_state():
        return runner.snapshot()

    @app.get("/api/logs")
    def list_logs():
        d = proj().logs
        files = sorted(d.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True) if d.exists() else []
        return [{"name": f.name, "size": f.stat().st_size,
                 "mtime": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")} for f in files]

    @app.get("/api/logs/{file}")
    def get_log(file: str, n: int = 200):
        d = proj().logs
        if not LOG_NAME.match(file):
            raise HTTPException(400, "로그 파일 이름이 잘못됨")
        f = d / file
        if not _within(f, d):
            raise HTTPException(400, "로그 폴더 밖")
        if not f.is_file():
            raise HTTPException(404, "로그 없음")
        return {"name": file, "lines": _tail(f, max(1, min(n, 5000)))}

    # ---------- 결과 ----------
    @app.get("/api/report", response_class=HTMLResponse)
    def get_report(rebuild: int = 0):
        p = proj()
        if rebuild or not p.report.exists():
            from ..stages.report import write_report
            write_report(p)
        return HTMLResponse(p.report.read_text(encoding="utf-8"))

    # ---------- 화면 ----------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _range_response(path: Path, range_header: str | None) -> Response:
    size = path.stat().st_size
    ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
    headers = {"Accept-Ranges": "bytes"}
    if not range_header:
        return FileResponse(path, media_type=ctype, headers=headers)
    m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if m.group(1) == "":                      # 끝에서 n 바이트
        start = max(0, size - int(m.group(2)))
        end = size - 1
    else:
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    headers |= {"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)}
    return StreamingResponse(body(), status_code=206, media_type=ctype, headers=headers)


def serve(project_root: str | Path, port: int = 8765, open_browser: bool = False) -> None:
    import uvicorn
    app = create_app(project_root)
    url = f"http://127.0.0.1:{port}/"
    print(f"IndoorGS GUI: {url}  (끄기: Ctrl+C)")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
