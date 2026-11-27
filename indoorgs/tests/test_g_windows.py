"""G: windows/*.bat 검사 (리눅스에서는 실행 못 하니 형식·경로만).
- ASCII 만: 한글이 있으면 cmd 가 CP949 로 잘못 읽어 줄이 실행 안 됨 (3_train/PROGRESS.md 시행착오)
- CRLF 줄바꿈: LF 만이면 cmd 가 goto 라벨·긴 줄을 잘못 읽을 수 있음
- 참조 경로가 저장소 구조와 맞음
"""
import re
from pathlib import Path

import pytest

from indoorgs import cli

WIN = Path(__file__).resolve().parents[1] / "windows"
BATS = ["install.bat", "IndoorGS.bat", "new_project.bat"]


@pytest.mark.parametrize("name", BATS)
def test_ascii_crlf(name):
    raw = (WIN / name).read_bytes()
    raw.decode("ascii")
    assert raw.count(b"\n") == raw.count(b"\r\n") > 5
    assert raw.endswith(b"\r\n")
    assert raw.lstrip().lower().startswith(b"@echo off")


@pytest.mark.parametrize("name", BATS)
def test_labels_exist(name):
    text = (WIN / name).read_text(encoding="ascii")
    labels = {m.lower() for m in re.findall(r"^:(\w+)", text, re.M)}
    for target in re.findall(r"goto :?(\w+)", text, re.I):
        assert target.lower() == "eof" or target.lower() in labels, f"{name}: goto {target} 라벨 없음"


def _resolve(expr: str) -> Path:
    """'%~dp0..\\..' 같은 식을 windows 폴더 기준 실제 경로로."""
    rest = expr.replace("%~dp0", "").replace("\\", "/").strip("/")
    return (WIN / rest).resolve() if rest else WIN


def test_paths_match_repo():
    pkg = WIN.parent
    repo = pkg.parent
    inst = (WIN / "install.bat").read_text(encoding="ascii")
    assert '("%~dp0..")' in inst and '("%PKG%\\..")' in inst
    assert _resolve("%~dp0..") == pkg and (pkg / "pyproject.toml").exists()
    assert '"%PKG%[all]"' in inst and 'VENV=%REPO%\\.venv' in inst
    for n in ("IndoorGS.bat", "new_project.bat"):
        t = (WIN / n).read_text(encoding="ascii")
        m = re.search(r'for %%I in \("(%~dp0[^"]*)"\) do set "REPO=', t)
        assert m and _resolve(m.group(1)) == repo, n
        assert "\\.venv\\Scripts\\indoorgs.exe" in t
    assert '"%~dp0IndoorGS.bat"' in (WIN / "new_project.bat").read_text(encoding="ascii")


def test_all_extra_has_gui_space_fill():
    tomllib = pytest.importorskip("tomllib")   # 3.11+
    data = tomllib.loads((WIN.parent / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert extras["all"] == ["indoorgs[gui,space,fill]"]
    assert data["project"]["scripts"]["indoorgs"] == "indoorgs.cli:main"


def test_bat_commands_exist_in_cli():
    """bat 이 부르는 명령·옵션이 cli 에 있음."""
    import argparse
    calls = []
    orig = argparse.ArgumentParser.parse_args

    def spy(self, args=None, ns=None):
        a = orig(self, args, ns)
        calls.append(a)
        raise SystemExit(0)
    for argv in (["gui", "X", "--open"], ["init", "X", "--search", "F:/a", "D:/b"], ["check", "X", "--search", "F:/a"]):
        argparse.ArgumentParser.parse_args = spy
        try:
            with pytest.raises(SystemExit):
                cli.main(argv)
        finally:
            argparse.ArgumentParser.parse_args = orig
    assert calls[0].open is True and calls[1].search == ["F:/a", "D:/b"] and calls[2].search == ["F:/a"]
    text = (WIN / "new_project.bat").read_text(encoding="ascii")
    assert re.search(r'init "%TARGET%" --search %SEARCH%', text)
    assert re.search(r'gui "%PROJ%" --open', (WIN / "IndoorGS.bat").read_text(encoding="ascii"))
