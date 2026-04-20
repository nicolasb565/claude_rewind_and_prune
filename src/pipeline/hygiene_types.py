"""Shared types for the hygiene-training data pipeline.

A Session is a normalized view of a Claude Code trajectory. Checkpoints are
annotations saying "fire checkpoint_progress here with this summary."

Normalization intent: one Step per completed tool_use+tool_result pair plus
short text turns, so position-references are simple integer indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Any

ProgressType = Literal["milestone_achieved", "approach_eliminated"]


@dataclass
class Step:
    """One tool call + result, or a text-only turn."""

    idx: int
    role: Literal["user_text", "assistant_text", "tool"]
    tool_name: str = ""           # Bash, Read, Edit, etc. empty for text turns
    cmd: str = ""                  # primary input (command, path, pattern…)
    output: str = ""               # tool_result text
    input_file: str | None = None  # file_path from input when present
    text: str = ""                 # for text turns — user prompt or assistant prose

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Checkpoint:
    """An annotation saying 'fire checkpoint_progress between step after_step and after_step+1'."""

    after_step: int
    progress_type: ProgressType
    finding: str
    evidence: str
    next_direction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    session_id: str
    source: str                    # dataset name + row id
    steps: list[Step] = field(default_factory=list)
    # Annotations stored as an ordered event stream — each event is either
    # {"expire": step_id} or {"checkpoint": {Checkpoint-fields}}. Store as the
    # canonical form because the training renderer needs the events in
    # step order; grouped views are derived on demand.
    events: list[dict[str, Any]] = field(default_factory=list)

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def checkpoints(self) -> list[Checkpoint]:
        out = []
        for ev in self.events:
            cp = ev.get("checkpoint") if isinstance(ev, dict) else None
            if isinstance(cp, dict):
                out.append(Checkpoint(**cp))
        return out

    @property
    def expire_step_ids(self) -> list[int]:
        return [int(ev["expire"]) for ev in self.events
                if isinstance(ev, dict) and "expire" in ev]

    def set_annotations(self, checkpoints: list[Checkpoint], expire_ids: list[int]) -> None:
        """Merge grouped annotations into an ordered event stream."""
        events: list[dict[str, Any]] = []
        events.extend({"expire": int(i)} for i in expire_ids)
        events.extend({"checkpoint": c.to_dict()} for c in checkpoints)

        def order_key(ev: dict[str, Any]) -> int:
            if "expire" in ev:
                return int(ev["expire"])
            return int(ev["checkpoint"]["after_step"])

        events.sort(key=order_key)
        self.events = events

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "steps": [s.to_dict() for s in self.steps],
            "events": list(self.events),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Session:
        events = list(d.get("events", []))
        # Backward-compat: upgrade older grouped-form payloads.
        if not events and ("checkpoints" in d or "expire_step_ids" in d):
            for i in d.get("expire_step_ids", []) or []:
                events.append({"expire": int(i)})
            for c in d.get("checkpoints", []) or []:
                events.append({"checkpoint": dict(c)})
            events.sort(key=lambda ev: int(ev.get("expire", ev.get("checkpoint", {}).get("after_step", 0))))
        return cls(
            session_id=d["session_id"],
            source=d["source"],
            steps=[Step(**s) for s in d["steps"]],
            events=events,
        )


# ── Compression helpers shared by renderers ───────────────────────────────

def truncate_tool_output(text: str, keep_first: int = 15, keep_last: int = 10) -> str:
    """Shrink large tool outputs for prompt-budget purposes.

    Keeps the first N and last M lines with an elision marker in between.
    Short outputs pass through unchanged.
    """
    if not text:
        return ""
    lines = text.split("\n")
    if len(lines) <= keep_first + keep_last + 3:
        return text
    elided = len(lines) - keep_first - keep_last
    kept = lines[:keep_first] + [f"[... {elided} lines omitted ...]"] + lines[-keep_last:]
    return "\n".join(kept)


def render_step_for_prompt(step: Step, max_output_chars: int = 2000) -> str:
    """Render a step into a compact human-readable form for inclusion in a prompt."""
    if step.role == "user_text":
        snippet = step.text[:500]
        tail = f"  (…{len(step.text) - 500} more chars)" if len(step.text) > 500 else ""
        return f"[{step.idx:3d}] USER: {snippet}{tail}"
    if step.role == "assistant_text":
        snippet = step.text[:500]
        tail = f"  (…{len(step.text) - 500} more chars)" if len(step.text) > 500 else ""
        return f"[{step.idx:3d}] ASST: {snippet}{tail}"
    # tool step
    cmd = step.cmd[:200]
    cmd_tail = f" (…{len(step.cmd) - 200})" if len(step.cmd) > 200 else ""
    out = truncate_tool_output(step.output)
    if len(out) > max_output_chars:
        out = out[:max_output_chars] + f"\n[... {len(out) - max_output_chars} chars truncated ...]"
    return f"[{step.idx:3d}] TOOL {step.tool_name}: {cmd}{cmd_tail}\n     → {out}"


def render_session_for_prompt(session: Session, max_output_chars: int = 2000) -> str:
    return "\n".join(render_step_for_prompt(s, max_output_chars) for s in session.steps)
