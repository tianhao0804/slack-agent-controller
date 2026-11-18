"""cwd validation keeps the caller's (symlinked) spelling but checks the real path."""

from __future__ import annotations

from pathlib import Path

from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


def test_validate_cwd_symlink(tmp_path: Path):
    home = tmp_path / "home"
    scratch = tmp_path / "scratch"
    outside = tmp_path / "outside"
    for d in (home, scratch / "proj", outside):
        d.mkdir(parents=True)
    (home / "proj").symlink_to(scratch / "proj")
    (home / "escape").symlink_to(outside)
    hub = Hub(make_config(tmp_path, roots=[home, scratch]), FakeWeb())  # type: ignore[arg-type]

    p, err = hub.validate_cwd(str(home / "proj"))
    assert err == "" and p == home / "proj"          # spelling kept for resume
    p, err = hub.validate_cwd("proj")                 # relative to first root
    assert err == "" and p == home / "proj"
    p, err = hub.validate_cwd(str(home / "escape"))   # symlink out of the roots
    assert p is None and "outside the allowed roots" in err
    p, err = hub.validate_cwd(str(home / "missing"))
    assert p is None and "not an existing directory" in err
