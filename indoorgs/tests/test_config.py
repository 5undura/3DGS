from pathlib import Path

import pytest

from indoorgs import config

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo"


def test_demo_loads():
    cfg = config.load(DEMO / "project.yaml")
    assert [s.name for s in cfg.sections] == ["corridor1", "room_b117"]
    assert cfg.train.trainer == "brush" and cfg.fill.holes_only


@pytest.mark.parametrize("bad, msg", [
    ({"sections": []}, "비어"),
    ({"sections": [{"name": "복도", "video": "a"}]}, "영문"),
    ({"sections": [{"name": "a", "video": "x"}, {"name": "a", "video": "y"}]}, "중복"),
    ({"sections": [{"name": "a", "video": "x"}], "train": {"trainer": "postshot"}}, "train.trainer"),
    ({"sections": [{"name": "a", "video": "x"}], "trian": {}}, "모르는 최상위"),
    ({"sections": [{"name": "a", "video": "x"}], "extract": {"min_dif": 3}}, "모르는 항목"),
])
def test_bad_configs(bad, msg):
    with pytest.raises(config.ConfigError, match=msg):
        config.parse(bad)
