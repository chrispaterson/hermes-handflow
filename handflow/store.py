"""Handflow state — the plan files the agent and the desktop pane both read.

Flow design ported from OpenManus's ``PlanningFlow`` (MIT, MetaGPT /
FoundationAgents): a plan is an ordered list of steps, each carrying a status
(``not_started`` / ``in_progress`` / ``completed`` / ``blocked``) and a note,
which the agent rewrites while it works. What differs from OpenManus is what is
*not* here: no LLM client, no tool collection, no browser. Hermes already has
the model and the tools — Handflow only keeps the plan, so a long task has one
visible spine instead of a scrollback.

Layout under ``$HERMES_HOME/handflow`` (override with ``HANDFLOW_HOME``)::

    plans/<plan_id>.json    one file per plan
    active.json             {"plan_id": "..."} — the profile-wide active plan
    active/<hash>.json      {"plan_id": "...", "key": "..."} — one per scope key

Which "active" pointer a call uses depends on the ``plan_scope`` setting:

* ``profile`` (default) — one active plan per profile, ``active.json``. This is
  the original behaviour.
* ``session`` — one active plan per Hermes session.
* ``project`` — one active plan per git checkout (worktree root of the
  session's cwd), so a session and the subagents it spawns share one plan.
  A session whose cwd is not inside a git checkout falls back to ``session``.

Stdlib only. Every write is atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
BLOCKED = "blocked"

STATUSES = (NOT_STARTED, IN_PROGRESS, COMPLETED, BLOCKED)

# Same marker vocabulary as OpenManus, so the rendered plan reads the same way.
MARKS = {
    COMPLETED: "[✓]",
    IN_PROGRESS: "[→]",
    BLOCKED: "[!]",
    NOT_STARTED: "[ ]",
}

MAX_EVENTS = 200

# Status → the act that produced it, so the event trail speaks the tool's
# vocabulary ("complete", not "completed").
_VERBS = {
    IN_PROGRESS: "begin",
    COMPLETED: "complete",
    BLOCKED: "block",
    NOT_STARTED: "reopen",
}

PLANS_SUBDIR = "plans"
ACTIVE_FILE = "active.json"
ACTIVE_SUBDIR = "active"

SCOPE_PROFILE = "profile"
SCOPE_SESSION = "session"
SCOPE_PROJECT = "project"
SCOPES = (SCOPE_PROFILE, SCOPE_SESSION, SCOPE_PROJECT)

PLUGIN_ID = "handflow"


# --------------------------------------------------------------------------- #
# paths (resolved per call, never cached: one process serves several profiles)
# --------------------------------------------------------------------------- #


def state_root() -> Path:
    """Return the state directory, honouring ``HANDFLOW_HOME`` then ``HERMES_HOME``."""
    override = os.environ.get("HANDFLOW_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    home = os.environ.get("HERMES_HOME", "").strip() or "~/.hermes"
    return Path(home).expanduser() / "handflow"


def plans_dir() -> Path:
    return state_root() / PLANS_SUBDIR


def active_path(key: str = "") -> Path:
    """The pointer file for ``key``; the empty key is the profile-wide ``active.json``."""
    if not key:
        return state_root() / ACTIVE_FILE
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return state_root() / ACTIVE_SUBDIR / f"{digest}.json"


# --------------------------------------------------------------------------- #
# scope: which "active plan" a caller sees
# --------------------------------------------------------------------------- #


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "").strip() or "~/.hermes").expanduser()


def _configured_scope() -> str:
    """``plugins.entries.handflow.settings.plan_scope`` from the Hermes config, if readable.

    Read per call (the setting can change while Hermes runs) and only when the
    Hermes config loader is importable, so the store still works standalone.
    """
    try:
        from hermes_cli.config import load_config_readonly  # type: ignore
    except Exception:
        return ""
    try:
        config = load_config_readonly() or {}
    except Exception:
        return ""
    entry = ((config.get("plugins") or {}).get("entries") or {}).get(PLUGIN_ID) or {}
    for subtree in ("settings", "config"):
        value = (entry.get(subtree) or {}).get("plan_scope") if isinstance(entry, dict) else None
        if value:
            return str(value)
    return ""


def plan_scope() -> str:
    """The configured scope: ``HANDFLOW_SCOPE`` env, then plugin settings, then ``profile``."""
    raw = os.environ.get("HANDFLOW_SCOPE", "").strip() or _configured_scope()
    scope = raw.strip().lower()
    return scope if scope in SCOPES else SCOPE_PROFILE


def session_dir(session_id: str) -> str:
    """The session's working directory from Hermes' session store (read-only)."""
    db = _hermes_home() / "state.db"
    if not session_id or not db.is_file():
        return ""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute("SELECT cwd FROM sessions WHERE id = ?", (session_id,)).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    return str(row[0] or "") if row else ""


def _is_checkout(path: Path) -> bool:
    """A real git checkout: ``.git/HEAD`` in a clone, a ``gitdir:`` file in a worktree."""
    marker = path / ".git"
    try:
        if marker.is_dir():
            return (marker / "HEAD").is_file()
        if marker.is_file():
            return marker.read_text(encoding="utf-8", errors="replace").startswith("gitdir:")
    except OSError:
        return False
    return False


def checkout_root(path: str) -> str:
    """Nearest ancestor that is a git checkout, stopping below the home directory.

    ``$HOME`` itself never counts: Desktop sessions start there, and a dotfiles
    repo (or a stray ``~/.git``) would otherwise put every such session on one plan.
    """
    if not path:
        return ""
    try:
        current = Path(path).expanduser().resolve()
        home = Path.home().resolve()
    except OSError:
        return ""
    for candidate in (current, *current.parents):
        if candidate == home or candidate == candidate.parent:
            return ""
        if _is_checkout(candidate):
            return str(candidate)
    return ""


def scope_key(session_id: str = "", scope: str = "") -> str:
    """The pointer key for a caller. Empty means the profile-wide pointer.

    No session id (CLI, tests, an older Hermes) always resolves to the profile
    pointer, so the tool keeps working wherever it runs.
    """
    scope = scope or plan_scope()
    session_id = str(session_id or "").strip()
    if scope == SCOPE_PROFILE or not session_id:
        return ""
    if scope == SCOPE_PROJECT:
        root = checkout_root(session_dir(session_id))
        if root:
            return f"project:{root}"
    return f"session:{session_id}"


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def pin(plan_id: str, key: str, session_id: str = "") -> None:
    """Make ``plan_id`` active for ``key``. A project key also pins the session's own
    pointer, so the plan survives the session changing checkout (see ``resolve``)."""
    set_active(plan_id, key)
    session_id = str(session_id or "").strip()
    if session_id and key.startswith("project:"):
        set_active(plan_id, _session_key(session_id))


def resolve(session_id: str = "", scope: str = "") -> "tuple[str, Optional[Dict[str, Any]]]":
    """``(key, active plan)`` for a caller.

    Under project scope the key follows the session's *current* checkout. A session
    that started its plan in one checkout and then moved (a ``cd``, ``desktop_project``)
    would lose it, so when the checkout has no plan of its own, fall back to the
    session's pointer and re-pin the new checkout to that plan (subagents working
    there then share it). A checkout that already has a plan is never taken over.
    """
    key = scope_key(session_id, scope)
    plan = active_plan(key)
    session_id = str(session_id or "").strip()
    if plan is None and session_id and key.startswith("project:"):
        plan = active_plan(_session_key(session_id))
        if plan is not None and plan.get("status") == "active":
            set_active(plan["id"], key)
    return key, plan


def plan_path(plan_id: str) -> Path:
    safe = "".join(c for c in plan_id if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("invalid plan id")
    return plans_dir() / f"{safe}.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def close_if_done(plan: Dict[str, Any]) -> bool:
    """Every step complete ⇒ the plan closes itself (add/begin reopens it).

    Shared by the agent tool and the backend route: the pane clicking "complete"
    on the last step must close the plan exactly like the agent's own call does.
    """
    tally = counts(plan)
    if plan.get("status") == "active" and tally["total"] and tally["done"] == tally["total"]:
        plan["status"] = "done"
        plan["closed_at"] = _now()
        _log_event(plan, "auto_finish", "all steps complete")
        return True
    return False


def reopen(plan: Dict[str, Any]) -> None:
    """Un-close a finished plan (a new step, or a step moving back to not_started)."""
    if plan.get("status") != "active":
        plan["status"] = "active"
        plan["closed_at"] = ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #


def new_plan_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"plan_{int(time.time())}_{suffix}"


def make_step(title: str, executor: str = "", position: int = 1) -> Dict[str, Any]:
    return {
        # ids are positional (s01…) so they stay unique and readable in tool
        # output; step resolution by 1-based number works either way.
        "id": f"s{position:02d}",
        "title": title.strip(),
        "status": NOT_STARTED,
        "notes": [],
        "executor": executor.strip(),
        "created_at": _now(),
        "closed_at": "",
    }


def relabel(plan: Dict[str, Any]) -> None:
    """Renumber step ids to match position — after an insert, the plan still reads
    as one ordered list (s01, s02, …) instead of ids that skip or collide."""
    for i, step in enumerate(plan.get("steps") or []):
        step["id"] = f"s{i + 1:02d}"


def new_plan(goal: str, step_titles: Iterable[str], executor: str = "", plan_id: str = "") -> Dict[str, Any]:
    """Build (but do not save) a plan whose first step is already in progress."""
    steps = [make_step(t, executor, i + 1) for i, t in enumerate(step_titles) if str(t).strip()]
    if not steps:
        raise ValueError("a plan needs at least one step")
    plan = {
        "id": plan_id or new_plan_id(),
        "goal": goal.strip(),
        "status": "active",
        "executor": executor.strip(),
        "steps": steps,
        "events": [],
        "created_at": _now(),
        "updated_at": _now(),
        "closed_at": "",
    }
    relabel(plan)
    steps[0]["status"] = IN_PROGRESS
    steps[0]["notes"].append("started")
    _log_event(plan, "start", f"{len(steps)} steps")
    return plan


def _log_event(plan: Dict[str, Any], kind: str, detail: str = "") -> None:
    plan.setdefault("events", []).append({"ts": _now(), "kind": kind, "detail": detail})
    if len(plan["events"]) > MAX_EVENTS:
        del plan["events"][: len(plan["events"]) - MAX_EVENTS]


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def save(plan: Dict[str, Any]) -> Path:
    plan["updated_at"] = _now()
    path = plan_path(plan["id"])
    _write_json(path, plan)
    return path


def load(plan_id: str) -> Optional[Dict[str, Any]]:
    return _read_json(plan_path(plan_id))


def set_active(plan_id: str, key: str = "") -> None:
    payload = {"plan_id": plan_id, "at": _now()}
    if key:
        payload["key"] = key
    _write_json(active_path(key), payload)


def active_plan_id(key: str = "") -> str:
    payload = _read_json(active_path(key)) or {}
    return str(payload.get("plan_id") or "")


def active_plan(key: str = "") -> Optional[Dict[str, Any]]:
    """The active plan for ``key``.

    The profile-wide pointer falls back to the most recently updated plan when
    nothing is pinned. A scoped pointer never does: falling back there would
    show (and let the agent edit) a plan that belongs to another session.
    """
    plan_id = active_plan_id(key)
    if plan_id:
        plan = load(plan_id)
        if plan is not None:
            return plan
    if key:
        return None
    recent = list_plans(limit=1, include_closed=True)
    return recent[0] if recent else None


def list_plans(limit: int = 20, include_closed: bool = True) -> List[Dict[str, Any]]:
    """Plans newest-first. Corrupt or half-written files are skipped, not raised.

    "Newest" is the file's write time, not ``updated_at``: the timestamp is only
    second-resolution, so two plans created in the same second would otherwise
    order by their random id suffix — a flaky "most recent" that bit the
    no-active-pointer fallback. ``id`` stays as a deterministic tie-break.
    """
    out: List[tuple] = []
    for path in plans_dir().glob("*.json") if plans_dir().is_dir() else []:
        plan = _read_json(path)
        if not isinstance(plan, dict) or "id" not in plan:
            continue
        if not include_closed and plan.get("status") != "active":
            continue
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:  # vanished between glob and stat — skip it, don't raise
            continue
        out.append((stamp, str(plan.get("id") or ""), plan))
    out.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in out[: max(1, int(limit))]]


# --------------------------------------------------------------------------- #
# reading and mutating a plan
# --------------------------------------------------------------------------- #


def find_step(plan: Dict[str, Any], ref: Any) -> Optional[Dict[str, Any]]:
    """Resolve a step by 1-based position, by id, or by title substring."""
    steps = plan.get("steps") or []
    if ref in (None, "", "current", "next"):
        for step in steps:
            if step.get("status") == IN_PROGRESS:
                return step
        for step in steps:
            if step.get("status") == NOT_STARTED:
                return step
        return None
    text = str(ref).strip()
    if text.isdigit():
        index = int(text) - 1
        if 0 <= index < len(steps):
            return steps[index]
        return None
    for step in steps:
        if step.get("id") == text:
            return step
    needle = text.lower()
    for step in steps:
        if needle in str(step.get("title", "")).lower():
            return step
    return None


def mark_step(plan: Dict[str, Any], ref: Any, status: str, note: str = "") -> Dict[str, Any]:
    """Set a step's status (and append a note). Refuses unknown or invalid input."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    step = find_step(plan, ref)
    if step is None:
        raise ValueError(f"no step matches {ref!r}")
    step["status"] = status
    if status in (COMPLETED, BLOCKED):
        step["closed_at"] = _now()
    if status == IN_PROGRESS:
        step["closed_at"] = ""
    if note.strip():
        step["notes"].append(note.strip())
    if status == IN_PROGRESS:
        _auto_start_next(plan, step, note)
    # events read as the act performed, not as the resulting status word, so the
    # audit trail lines up with the tool's action vocabulary.
    _log_event(plan, _VERBS.get(status, status), f"{step.get('id')} {step.get('title', '')[:60]}")
    return step


def _auto_start_next(plan: Dict[str, Any], step: Dict[str, Any], note: str) -> None:
    """A step marked in progress reopens nothing, but any other in-progress step closes."""
    for other in plan.get("steps") or []:
        if other is step or other.get("status") != IN_PROGRESS:
            continue
        other["status"] = COMPLETED
        other["closed_at"] = _now()
        other["notes"].append(f"auto-completed when {step.get('id')} started")
        _log_event(plan, "auto_complete", str(other.get("id")))


def add_step(plan: Dict[str, Any], title: str, position: Optional[int] = None) -> Dict[str, Any]:
    """Insert a step (replanning). ``position`` is 1-based; default appends."""
    step = make_step(title)
    steps = plan.setdefault("steps", [])
    if position is None or not str(position).strip():
        steps.append(step)
    else:
        index = max(0, min(len(steps), int(position) - 1))
        steps.insert(index, step)
    # renumber so the list still reads s01, s02, … after the insert
    relabel(plan)
    _log_event(plan, "add_step", step["title"][:60])
    return step


def open_work(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan.get("steps") or [] if s.get("status") in (NOT_STARTED, IN_PROGRESS)]


def reindex(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in ids/notes for plans written by an older version. Never raises."""
    for i, step in enumerate(plan.get("steps") or []):
        step.setdefault("id", f"s{i + 1:02d}")
        step.setdefault("title", f"step {i + 1}")
        step.setdefault("status", NOT_STARTED)
        step.setdefault("notes", [])
        step.setdefault("executor", "")
        step.setdefault("created_at", plan.get("created_at", _now()))
        step.setdefault("closed_at", "")
        if step["status"] not in STATUSES:
            step["status"] = NOT_STARTED
        if isinstance(step.get("notes"), str):
            step["notes"] = [step["notes"]] if step["notes"] else []
    plan.setdefault("events", [])
    plan.setdefault("status", "active")
    return plan


def counts(plan: Dict[str, Any]) -> Dict[str, int]:
    steps = plan.get("steps") or []
    tally = {status: 0 for status in STATUSES}
    for step in steps:
        tally[step.get("status", NOT_STARTED)] = tally.get(step.get("status", NOT_STARTED), 0) + 1
    tally["total"] = len(steps)
    tally["done"] = tally[COMPLETED]
    return tally


def render(plan: Dict[str, Any], width: int = 0) -> str:
    """The plan as text, one line per step — what the model reads back."""
    if not plan:
        return "no plan"
    lines = [f"plan {plan.get('id')}  ({plan.get('status')})", f"goal: {plan.get('goal')}"]
    for i, step in enumerate(plan.get("steps") or [], start=1):
        mark = MARKS.get(step.get("status", NOT_STARTED), "[ ]")
        lines.append(f"{i}. {mark} {step.get('title', '')}")
        for note in step.get("notes") or []:
            lines.append(f"   note: {note}")
    tally = counts(plan)
    lines.append(f"{tally['done']}/{tally['total']} complete · {tally[BLOCKED]} blocked")
    if plan.get("status") != "active":
        lines.append(f"closed: {plan.get('closed_at')}")
    return "\n".join(lines)
