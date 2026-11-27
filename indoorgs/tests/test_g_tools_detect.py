"""G: 도구 자동 찾기 (가짜 폴더 트리, 리눅스에서도 돌아감)."""
import os
from pathlib import Path

import yaml

from indoorgs import cli, tools_detect as td


def touch(p: Path, mtime: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def fake_tools(root: Path) -> dict[str, Path]:
    """F:/Sample/tools 와 비슷한 구조."""
    return {
        "colmap": touch(root / "colmap" / "COLMAP.bat", 2000),
        "colmap_old": touch(root / "colmap_old" / "colmap-x64" / "bin" / "colmap.exe", 1000),
        "brush": touch(root / "brush" / "brush_app.exe", 1500),
        # LichtFeld: build 아래 하위 폴더 (깊이 4)
        "lfs": touch(root / "lfs" / "LichtFeld-Studio" / "build" / "Release" / "LichtFeld-Studio.exe", 3000),
        "lfs_shallow": touch(root / "lfs" / "LichtFeld-Studio" / "build" / "LichtFeld-Studio.exe", 2500),
    }


NO_ENV = {"PATH": ""}


def test_find_in_search_root_newest_first(tmp_path):
    t = fake_tools(tmp_path / "tools")
    found = td.find_tools([tmp_path / "tools"], env=NO_ENV)
    assert [c.path for c in found["colmap"]] == [str(t["colmap"]), str(t["colmap_old"])]
    assert td.best(found) == {"colmap": str(t["colmap"]), "brush": str(t["brush"]), "lichtfeld": str(t["lfs"])}
    assert len(found["lichtfeld"]) == 2 and all(c.source == "search" for c in found["lichtfeld"])


def test_depth_limit_and_skip_dirs(tmp_path):
    root = tmp_path / "tools"
    touch(root / "a" / "b" / "c" / "d" / "e" / "brush_app.exe")          # 폴더 깊이 5: 안 찾음
    touch(root / "lfs" / "vcpkg" / "x" / "LichtFeld-Studio.exe")          # vcpkg: 안 훑음
    touch(root / "colmap_skip" / "COLMAP.bat")                            # *_skip: 뺀 것
    found = td.find_tools([root], env=NO_ENV)
    assert found == {"colmap": [], "brush": [], "lichtfeld": []}
    touch(root / "a" / "b" / "c" / "d" / "brush_app.exe")                # 깊이 4: 찾음
    assert len(td.find_tools([root], env=NO_ENV)["brush"]) == 1


def test_case_insensitive_and_path(tmp_path):
    touch(tmp_path / "x" / "colmap.EXE")
    assert len(td.find_tools([tmp_path / "x"], env=NO_ENV)["colmap"]) == 1
    exe = touch(tmp_path / "bin" / "brush_app.exe")
    exe.chmod(0o755)
    found = td.find_tools([], env={"PATH": str(tmp_path / "bin")}, use_common=False)
    assert found["brush"][0].source == "path"


def test_common_roots_keyword_filter(tmp_path):
    local = tmp_path / "local"
    touch(local / "Programs" / "COLMAP" / "COLMAP.bat")                    # 'colmap' 이름 없는 상위: 안 들어감
    keep = touch(local / "colmap-4.2.1" / "COLMAP.bat")
    found = td.find_tools([], env={"PATH": "", "LOCALAPPDATA": str(local)})
    assert [c.path for c in found["colmap"]] == [str(keep)]
    assert found["colmap"][0].source == "common"


def test_no_duplicates_when_roots_overlap(tmp_path):
    fake_tools(tmp_path / "tools")
    found = td.find_tools([tmp_path / "tools", tmp_path / "tools" / "colmap"], env=NO_ENV)
    assert len(found["colmap"]) == 2


def test_yaml_path_roundtrip():
    for p in [r"F:\Sample\tools\colmap\COLMAP.bat", "C:/Program Files/it's #1/brush_app.exe"]:
        assert yaml.safe_load("a: " + td.yaml_path(p))["a"] == p.replace("\\", "/")


def test_init_fills_found_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", "")
    t = fake_tools(tmp_path / "tools")
    assert cli.main(["init", str(tmp_path / "proj"), "--search", str(tmp_path / "tools")]) == 0
    data = yaml.safe_load((tmp_path / "proj" / "project.yaml").read_text(encoding="utf-8"))
    assert data["tools"]["colmap"] == str(t["colmap"]).replace("\\", "/")
    assert data["tools"]["brush"] == str(t["brush"]).replace("\\", "/")
    assert data["tools"]["lichtfeld"] == str(t["lfs"]).replace("\\", "/")
    assert "찾음" in capsys.readouterr().out


def test_init_without_tools_keeps_template(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert cli.main(["init", str(tmp_path / "proj"), "--search", str(tmp_path / "empty")]) == 0
    data = yaml.safe_load((tmp_path / "proj" / "project.yaml").read_text(encoding="utf-8"))
    assert data["tools"]["lichtfeld"] is None and data["tools"]["colmap"].endswith("COLMAP.bat")


def test_check_lists_other_candidates(tmp_path, monkeypatch, capsys):
    from indoorgs.colmap import runner
    from indoorgs.colmap.runner import ToolResult
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(runner, "run", lambda cmd, **k: ToolResult(0, ["COLMAP 4.2.1", "pose_prior_mapper"]))
    t = fake_tools(tmp_path / "tools")
    proj = tmp_path / "proj"
    cli.main(["init", str(proj), "--search", str(tmp_path / "tools")])
    capsys.readouterr()
    cli.main(["check", str(proj)])     # --search 없이도 설정 경로의 두 단계 위(tools)를 훑음
    out = capsys.readouterr().out
    assert f"자동으로 찾은 다른 후보 colmap   : {t['colmap_old']}" in out
    assert str(t["lfs_shallow"]) in out
    assert f"후보 colmap   : {t['colmap']}" not in out       # 설정된 것은 안 보여 줌
