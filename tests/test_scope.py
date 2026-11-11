"""plan_scope: which sessions share an active plan.

The default (``profile``) must behave exactly like before the setting existed;
``session`` and ``project`` give parallel sessions their own plans.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from handflow import store
from handflow.tools import handflow


def run(session_id: str = "", **args):
    """Call the tool the way the registry does: model args positional, context as kwargs."""
    extra = {"session_id": session_id} if session_id else {}
    return json.loads(handflow(dict(args), task_id="t", **extra))


@pytest.fixture()
def sessions(tmp_path, monkeypatch, hf_home):
    """A fake Hermes session store; returns ``add(session_id, cwd)``."""
    hermes = tmp_path / "hermes"
    hermes.mkdir(exist_ok=True)
    db = sqlite3.connect(hermes / "state.db")
    db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT)")
    db.commit()

    def add(session_id: str, cwd) -> None:
        db.execute("INSERT INTO sessions VALUES (?, ?)", (session_id, str(cwd) if cwd is not None else None))
        db.commit()

    yield add
    db.close()


def checkout(path: Path, worktree: bool = False) -> Path:
    """A git clone (``.git`` dir) or a linked worktree (``.git`` file)."""
    path.mkdir(parents=True)
    if worktree:
        (path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    else:
        (path / ".git").mkdir()
        (path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return path


def test_home_directory_is_never_a_project(hf_home, monkeypatch, sessions, tmp_path):
    """Regression: a stray ~/.git (or a dotfiles repo) put every $HOME session on one plan."""
    monkeypatch.setenv("HANDFLOW_SCOPE", "project")
    home = checkout(tmp_path / "home")
    monkeypatch.setenv("HOME", str(home))
    (home / "notes").mkdir()
    sessions("h1", home)
    sessions("h2", home / "notes")
    assert store.scope_key("h1") == "session:h1"
    assert store.scope_key("h2") == "session:h2"


def test_an_invalid_git_dir_is_not_a_checkout(tmp_path):
    (tmp_path / "x" / ".git").mkdir(parents=True)
    assert store.checkout_root(str(tmp_path / "x")) == ""


# --------------------------------------------------------------------------- #
# the setting itself
# --------------------------------------------------------------------------- #


def test_default_scope_is_profile(hf_home):
    assert store.plan_scope() == "profile"


@pytest.mark.parametrize("raw, expected", [("session", "session"), ("PROJECT", "project"), ("bogus", "profile")])
def test_env_override_is_validated(hf_home, monkeypatch, raw, expected):
    monkeypatch.setenv("HANDFLOW_SCOPE", raw)
    assert store.plan_scope() == expected


def test_plugin_setting_is_used_when_env_is_unset(hf_home, monkeypatch):
    monkeypatch.setattr("handflow.store._configured_scope", lambda: "session")
    assert store.plan_scope() == "session"


def test_manifest_declares_the_setting():
    from helpers import plugin_yaml

    schema = plugin_yaml()["config_schema"]["plan_scope"]
    assert schema["default"] == "profile"


# --------------------------------------------------------------------------- #
# profile scope: unchanged behaviour
# --------------------------------------------------------------------------- #


def test_profile_scope_shares_one_plan_across_sessions(hf_home):
    a = run("sess-a", action="start", goal="A", steps=["a1"])
    b = run("sess-b", action="start", goal="B", steps=["b1"])
    assert run("sess-a", action="plan")["plan_id"] == b["plan_id"] != a["plan_id"]
    assert store.active_path("").name == "active.json"
    assert store.active_plan_id() == b["plan_id"]


# --------------------------------------------------------------------------- #
# session scope
# --------------------------------------------------------------------------- #


def test_session_scope_keeps_parallel_plans_apart(hf_home, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    a = run("sess-a", action="start", goal="A", steps=["a1", "a2"])
    b = run("sess-b", action="start", goal="B", steps=["b1"])

    done = run("sess-a", action="complete", note="a1 done")
    assert done["plan_id"] == a["plan_id"]

    plan_b = store.load(b["plan_id"])
    assert [s["status"] for s in plan_b["steps"]] == ["in_progress"]
    assert run("sess-b", action="plan")["plan_id"] == b["plan_id"]


def test_session_scope_never_falls_back_to_another_sessions_plan(hf_home, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    run("sess-a", action="start", goal="A", steps=["a1"])
    reply = run("sess-b", action="complete")
    assert reply["ok"] is False


def test_no_session_id_uses_the_profile_pointer(hf_home, monkeypatch):
    """CLI and older Hermes builds pass no session id; the tool must still work."""
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    reply = run(action="start", goal="g", steps=["one"])
    assert store.active_plan_id() == reply["plan_id"]


def test_model_cannot_pick_a_session_through_its_arguments(hf_home, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    a = run("sess-a", action="start", goal="A", steps=["a1"])
    reply = json.loads(handflow({"action": "plan", "session_id": "sess-a"}, session_id="sess-b"))
    assert reply.get("plan_id") != a["plan_id"]


def test_start_records_the_scope_key_on_the_plan(hf_home, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    a = run("sess-a", action="start", goal="A", steps=["a1"])
    assert store.load(a["plan_id"])["scope_key"] == "session:sess-a"


# --------------------------------------------------------------------------- #
# project scope
# --------------------------------------------------------------------------- #


def test_project_scope_shares_a_plan_inside_one_checkout(hf_home, monkeypatch, sessions, tmp_path):
    monkeypatch.setenv("HANDFLOW_SCOPE", "project")
    repo = checkout(tmp_path / "GRAPH-1", worktree=True)
    (repo / "packages" / "x").mkdir(parents=True)
    sessions("parent", repo)
    sessions("subagent", repo / "packages" / "x")

    started = run("parent", action="start", goal="G", steps=["one", "two"])
    assert run("subagent", action="complete")["plan_id"] == started["plan_id"]
    assert store.load(started["plan_id"])["scope_key"] == f"project:{repo.resolve()}"


def test_project_scope_keeps_worktrees_apart(hf_home, monkeypatch, sessions, tmp_path):
    monkeypatch.setenv("HANDFLOW_SCOPE", "project")
    sessions("s1", checkout(tmp_path / "GRAPH-1", worktree=True))
    sessions("s2", checkout(tmp_path / "GRAPH-2", worktree=True))

    one = run("s1", action="start", goal="one", steps=["a"])
    two = run("s2", action="start", goal="two", steps=["b"])
    assert run("s1", action="plan")["plan_id"] == one["plan_id"]
    assert run("s2", action="plan")["plan_id"] == two["plan_id"]


@pytest.mark.parametrize("cwd", ["home", None, ""])
def test_project_scope_falls_back_to_session_outside_a_checkout(hf_home, monkeypatch, sessions, tmp_path, cwd):
    """A Desktop session still sitting in $HOME (or with no cwd) must not share a plan."""
    monkeypatch.setenv("HANDFLOW_SCOPE", "project")
    home = tmp_path / "home"
    home.mkdir()
    value = home if cwd == "home" else cwd
    sessions("h1", value)
    sessions("h2", value)

    one = run("h1", action="start", goal="one", steps=["a"])
    two = run("h2", action="start", goal="two", steps=["b"])
    assert store.scope_key("h1") == "session:h1"
    assert run("h1", action="plan")["plan_id"] == one["plan_id"] != two["plan_id"]


def test_project_scope_survives_a_missing_session_store(hf_home, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "project")
    assert store.scope_key("nowhere") == "session:nowhere"


# --------------------------------------------------------------------------- #
# the pane's routes
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client(hf_home):
    pytest.importorskip("fastapi")
    import importlib.util
    import sys

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("handflow_api_scope", root / "dashboard" / "plugin_api.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["handflow_api_scope"] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix="/api/plugins/handflow")
    return TestClient(app)


def test_pane_shows_the_focused_sessions_plan(client, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    a = run("sess-a", action="start", goal="A", steps=["a1"])
    b = run("sess-b", action="start", goal="B", steps=["b1"])
    get = lambda sid: client.get("/api/plugins/handflow/active", params={"session_id": sid}).json()
    assert get("sess-a")["plan"]["id"] == a["plan_id"]
    assert get("sess-b")["plan"]["id"] == b["plan_id"]
    assert get("sess-c")["plan"] is None


def test_pane_pin_lands_on_the_focused_session(client, monkeypatch):
    monkeypatch.setenv("HANDFLOW_SCOPE", "session")
    a = run("sess-a", action="start", goal="A", steps=["a1"])
    run("sess-b", action="start", goal="B", steps=["b1"])
    client.post("/api/plugins/handflow/active", json={"plan_id": a["plan_id"], "session_id": "sess-b"})
    assert run("sess-b", action="plan")["plan_id"] == a["plan_id"]
