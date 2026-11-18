"""Protected files named through braces, wildcards, ~ and $HOME are seen by the checks;
in auto mode such commands go to Slack instead of the classifier."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from slack_agent_controller import policy
from slack_agent_controller.allowlist import Allowlist
from slack_agent_controller.hub import Hub
from tests.helpers import FakeWeb, make_config


# -- brace expansion ------------------------------------------------------------
@pytest.mark.parametrize("word,expected", [
    (".{e,}nv", [".env", ".nv"]),
    ("a{b,c{d,e}}f", ["abf", "acdf", "acef"]),
    ("{1..3}", ["1", "2", "3"]),
    ("{03..05}", ["03", "04", "05"]),
    ("{a..c}x", ["ax", "bx", "cx"]),
    ("{5..1..2}", ["5", "3", "1"]),
    ("{x,y}{1,2}", ["x1", "x2", "y1", "y2"]),
    ("{a{b,c}", ["{ab", "{ac"]),           # an unbalanced brace stays literal, the rest expands
])
def test_brace_expand(word, expected):
    assert policy.brace_expand(word) == expected


@pytest.mark.parametrize("word", ["{a}", "{}", "${HOME}/x", "{a,b", "x{y", "plain", "{1..}", "{a..bb}"])
def test_brace_expand_leaves_non_expansions_alone(word):
    assert policy.brace_expand(word) == [word]


def test_brace_expand_is_bounded():
    with pytest.raises(policy.TooManyAlternatives):
        policy.brace_expand("{1..100000}")
    with pytest.raises(policy.TooManyAlternatives):
        policy.brace_expand("{a,b}" * 10)       # 1024 results


# -- protected names after expansion ---------------------------------------------
@pytest.fixture
def proj(tmp_path: Path):
    work = tmp_path / "proj"
    (work / "sub").mkdir(parents=True)
    secret = tmp_path / "vault"
    secret.mkdir()
    (secret / "token.txt").write_text("x")
    for name in (".env", "key.pem", "notes.txt", "a.txt", "b.txt"):
        (work / name).write_text("x")
    cfg = make_config(tmp_path, roots=[tmp_path], sensitive=[secret])
    return cfg, work, secret


def flagged(cfg, cmd: str, cwd: Path) -> bool:
    return any("protected path" in w for w in policy.bash_warnings(cfg, cmd, cwd))


@pytest.mark.parametrize("cmd", [
    "cat .env", "cat ./.env", "cat sub/../.env", "cat .e*", "cat .e[n]v", "cat .{e,}nv", "cat .en?",
    "cat *.pem", "cat k*.pem", "cat {.env,notes.txt}", "grep -f .env notes.txt", "cat --file=.env",
    "head -n 3 -f.env", "cat notes.txt && cat .env", "cat < .env", "cat vault/../vault/tok*",
])
def test_protected_names_are_found_after_expansion(proj, cmd):
    cfg, work, secret = proj
    cmd = cmd.replace("vault/..", f"{secret}/..").replace("vault/tok*", f"{secret}/tok*")
    assert flagged(cfg, cmd, work), cmd


@pytest.mark.parametrize("cmd", [
    "cat notes.txt", "ls", "ls -la", "cat {a,b}.txt", "cat .nothing*", "echo hello", "cat sub/*", "cat a.*",
    "grep -rn TODO .", "python3 run.py --epochs 3", "nvidia-smi", "git status",
])
def test_ordinary_commands_are_not_flagged(proj, cmd):
    cfg, work, secret = proj
    assert not flagged(cfg, cmd, work), cmd


def test_unbounded_or_unresolvable_words_fail_closed(proj):
    cfg, work, _ = proj
    assert flagged(cfg, "cat {1..100000}", work)
    assert flagged(cfg, "cat a\x00b", work)


def test_star_does_not_match_dotfiles_like_the_shell(proj):
    cfg, _, _ = proj
    clean = proj[1].parent / "clean"
    clean.mkdir()
    (clean / ".env").write_text("x")
    (clean / "a.txt").write_text("x")
    assert not flagged(cfg, "cat *", clean)          # bash: * skips .env
    assert flagged(cfg, "cat .*", clean)             # but .* matches it
    assert flagged(cfg, "cat *", proj[1])            # here * does reach key.pem (not a dotfile)


def test_output_protected_uses_the_same_check(proj):
    cfg, work, _ = proj
    assert policy.output_protected(cfg, "Bash", {"command": "cat .e*"}, work)
    assert not policy.output_protected(cfg, "Bash", {"command": "cat notes.txt"}, work)


# -- Grep glob filter ------------------------------------------------------------
@pytest.mark.parametrize("glob_", [".env", ".e*", "*.pem", "{.env,x.py}", "id_rsa", "*", "src/**/.env"])
def test_grep_glob_that_selects_protected_files_goes_to_slack(proj, glob_):
    cfg, work, _ = proj
    inp = {"pattern": "x", "path": str(work), "glob": glob_}
    res = policy.decide(cfg, "Grep", inp, work)
    assert (res.decision, res.force_slack) == ("ask", True), glob_
    assert policy.output_protected(cfg, "Grep", inp, work)


@pytest.mark.parametrize("glob_", ["*.py", "*.txt", "src/**/*.py", "!.env", "*.py !*.pem", "{a,b}.txt", ""])
def test_ordinary_grep_globs_stay_allowed(proj, glob_):
    cfg, work, _ = proj
    inp = {"pattern": "x", "path": str(work), "glob": glob_}
    assert policy.decide(cfg, "Grep", inp, work).decision == "allow", glob_


def test_search_scope_containing_protected_paths_is_forced_to_slack(proj):
    cfg, work, secret = proj
    res = policy.decide(cfg, "Grep", {"pattern": "x", "path": str(secret.parent)}, work)
    assert (res.decision, res.force_slack) == ("ask", True)


# -- auto mode: the hook sends these to Slack, the rest to the classifier ---------------
def _auto_session(cfg, work: Path, mode: str):
    hub = Hub(cfg, FakeWeb())  # type: ignore[arg-type]
    sess = hub.create_session("D1", "1.0", work, "UOWNER")
    sess.rec.mode = mode
    return sess


def _hook(sess, tool: str, tin: dict, cwd: Path) -> dict:
    inp = {"tool_name": tool, "tool_input": tin, "cwd": str(cwd), "tool_use_id": "t1"}
    return asyncio.run(sess._pre_tool_hook(inp, "t1", None))


def _verdict(out: dict) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "deferred-to-classifier")


@pytest.mark.parametrize("cmd", ["cat .env", "cat .e*", "cat .{e,}nv", "cat *.pem", "cat {.env,notes.txt}"])
def test_auto_mode_asks_in_slack_for_protected_names(proj, cmd):
    cfg, work, _ = proj
    assert _verdict(_hook(_auto_session(cfg, work, "auto"), "Bash", {"command": cmd}, work)) == "ask", cmd


@pytest.mark.parametrize("cmd", ["cat notes.txt", "ls -la", "python3 run.py", "grep -rn TODO ."])
def test_auto_mode_defers_ordinary_commands_to_the_classifier(proj, cmd):
    cfg, work, _ = proj
    assert _verdict(_hook(_auto_session(cfg, work, "auto"), "Bash", {"command": cmd}, work)) \
        == "deferred-to-classifier", cmd


def test_ask_mode_still_asks_for_everything(proj):
    cfg, work, _ = proj
    sess = _auto_session(cfg, work, "ask")
    for cmd in ("cat notes.txt", "cat .e*"):
        assert _verdict(_hook(sess, "Bash", {"command": cmd}, work)) == "ask", cmd


def test_auto_mode_grep_with_protected_scope_asks_in_slack(proj):
    cfg, work, secret = proj
    sess = _auto_session(cfg, work, "auto")
    assert _verdict(_hook(sess, "Grep", {"pattern": "x", "path": str(secret.parent)}, work)) == "ask"
    assert _verdict(_hook(sess, "Grep", {"pattern": "x", "glob": ".env", "path": str(work)}, work)) == "ask"
    assert _verdict(_hook(sess, "Grep", {"pattern": "x", "glob": "*.py", "path": str(work)}, work)) == "allow"


def test_protected_names_are_never_remembered(proj):
    cfg, work, _ = proj
    al = Allowlist(cfg.state_dir / "allowlist.json", cfg)
    for cmd in ("cat .e*", "cat .{e,}nv", "cat .env"):
        assert al.learn({"command": cmd}, work, "UOWNER", "s")[0] is None, cmd
    assert al.learn({"command": "cat notes.txt"}, work, "UOWNER", "s")[0] is not None
