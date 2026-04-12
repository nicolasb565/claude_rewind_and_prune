"""Transcript formatter and label file writer for stuck-detection labeling."""

import json
import os
import tempfile
from datetime import datetime, timezone

SCHEMA_VERSION = 1
LABELER_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are labeling steps in a Claude Code session. Each step is one tool call.
Classify each step as P (productive), S (stuck), or U (unsure).

PRODUCTIVE: the session is making forward progress. Exploring a new approach,
writing code, reading a file for the first time, testing a hypothesis.
Errors are fine — what matters is that something new is being attempted.

STUCK: the session is in a loop. The same command, the same error, the same
edit repeated without a changed approach or new information. The work has
stopped moving forward.

UNSURE: genuine ambiguity that you cannot resolve from the transcript.
Use sparingly — not as a default.

Common patterns:
- First attempt at any command → P
- Same command, same error, second or third time → S
- Trying a different file, flag, or approach after failure → P
- Reading a file already read (same path appears earlier in the transcript) → S
- Tight compile/test loop with unchanged failure → S

Transition rules:
- The first step of a repeating pattern is still P; label S when repetition begins
- The first step after escaping a loop (new approach, new tool) is P again

Output: one label per step, comma-separated, nothing else.
Example: P,P,S,S,S,P,P,S,P"""

_MAX_OUTPUT_LEN = 500
_COMPACT_TEXT_LEN = 300

_TOOL_KEY_MAP = {
    "Glob": "pattern",
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "Bash": "command",
    "TodoWrite": "todos",
}


def _render_step(step: dict, index: int) -> str:
    """Render one step as a transcript block."""
    tool_name = step.get("tool_name", step.get("tool", "unknown"))
    cmd = step.get("cmd", "")
    output = step.get("output", "")

    # Build key argument line
    key_field = _TOOL_KEY_MAP.get(tool_name)
    if key_field:
        arg_line = f"{key_field}: {cmd}"
    elif cmd:
        arg_line = cmd
    else:
        inp_parts = []
        for k, v in step.items():
            if k not in ("tool", "tool_name", "output", "thinking", "file") and v:
                inp_parts.append(f"{k}: {v}")
        arg_line = ", ".join(inp_parts) if inp_parts else ""

    # Truncate output
    if len(output) > _MAX_OUTPUT_LEN:
        output_display = output[:_MAX_OUTPUT_LEN] + "[...]"
    else:
        output_display = output

    lines = [f"[{index}] {tool_name}"]
    if arg_line:
        lines.append(f"  {arg_line}")
    if output_display:
        lines.append(f"  → {output_display}")
    return "\n".join(lines)


def format_transcript(steps: list[dict]) -> tuple[str, int]:
    """Format a list of steps (and optional CompactBlocks) as a transcript.

    Args:
        steps: list of step dicts and/or CompactBlocks
               CompactBlocks have {"type": "compact", "text": str}

    Returns:
        (transcript_text, n_steps) where n_steps counts only actual tool call steps
    """
    blocks: list[str] = []
    step_index = 0

    for item in steps:
        if isinstance(item, dict) and item.get("type") == "compact":
            text = item.get("text", "")
            if len(text) > _COMPACT_TEXT_LEN:
                text = text[:_COMPACT_TEXT_LEN]
            blocks.append(f"[compact: {text}]")
        else:
            blocks.append(_render_step(item, step_index))
            step_index += 1

    n_steps = step_index
    transcript = "\n".join(blocks)
    transcript += f"\nTotal steps: {n_steps}"
    return transcript, n_steps


def validate_label_file(path: str, expected_n_steps: int) -> bool:
    """Validate a label file.

    Args:
        path: path to label file
        expected_n_steps: expected number of labels

    Returns:
        True if valid JSON with correct label count, False otherwise
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        labels = data.get("labels", [])
        return len(labels) == expected_n_steps
    except (json.JSONDecodeError, OSError, KeyError):
        return False


def write_label_file(
    path: str,
    session_id: str,
    source: str,
    labels: list[str],
    n_steps: int,
) -> None:
    """Write a label file atomically.

    Args:
        path: output path
        session_id: session identifier
        source: source name
        labels: list of label strings (PRODUCTIVE, STUCK, UNSURE)
        n_steps: number of steps (must match len(labels))
    """
    label_dir = os.path.dirname(path)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)
    data = {
        "session_id": session_id,
        "source": source,
        "n_steps": n_steps,
        "labeler": LABELER_MODEL,
        "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels": labels,
    }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=label_dir or ".", delete=False, suffix=".tmp", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


_LABEL_CHAR_MAP = {
    "p": "PRODUCTIVE",
    "s": "STUCK",
    "u": "UNSURE",
}


def parse_csv_labels(csv: str, n_steps: int) -> list[str]:
    """Parse a P,S,U,... CSV string into label strings.

    Args:
        csv: comma-separated labels (case-insensitive, whitespace stripped)
        n_steps: expected number of labels

    Returns:
        list of PRODUCTIVE/STUCK/UNSURE strings

    Raises:
        ValueError: if unknown characters found or count doesn't match
    """
    raw = csv.strip().rstrip(",")
    parts = [p.strip() for p in raw.split(",")]
    # Empty parts in the middle indicate a missing label, not a trailing comma.
    for i, p in enumerate(parts[:-1]):
        if not p:
            raise ValueError(f"empty label part at position {i}")
    # Filter any trailing empty string left after stripping all trailing commas
    parts = [p for p in parts if p]

    labels = []
    for part in parts:
        key = part.lower()
        if key not in _LABEL_CHAR_MAP:
            raise ValueError(f"Unknown label character: {part!r}")
        labels.append(_LABEL_CHAR_MAP[key])

    if len(labels) != n_steps:
        raise ValueError(f"Label count mismatch: got {len(labels)}, expected {n_steps}")

    return labels
