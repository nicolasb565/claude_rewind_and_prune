"""Extract per-step features from parsed session steps.

Feature computation logic is ported from abstract_trajectory.py and kept
self-contained here so the new pipeline has no dependency on the old src/.
"""

import json
import math
import os
import re
import tempfile
import zlib
from datetime import datetime, timezone

SCHEMA_VERSION = 3

STEP_FEATURES = [
    "tool_idx",
    "cmd_hash",
    "file_hash",
    "output_similarity",
    "has_prior_output",
    "output_length",
    "is_error",
    "step_index_norm",
]

_CRC32_NORM = 1.0 / (1 << 32)  # map uint32 → [0, 1)

TOOL_NAMES = ["bash", "edit", "view", "search", "create", "submit", "other"]
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOL_NAMES)}

MAX_OUTPUT_LINES = 100

_SILENT_CMD_RE = re.compile(
    r"^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b"
)
_FILE_EXT_RE = re.compile(r"\.[a-zA-Z]{1,5}$")
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.I
)

ERROR_PATTERNS = re.compile(
    r"(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied"
    r"|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError"
    r"|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError)",
    re.I,
)


def _cmd_semantic_key(cmd: str) -> str:
    if not cmd:
        return ""
    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not _SILENT_CMD_RE.match(p.strip())]
    if not real:
        return cmd.split()[0] if cmd.split() else ""
    first = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first.strip().split()
    if not tokens:
        return ""
    base = tokens[0].rsplit("/", 1)[-1]
    target = None
    for tok in tokens[1:]:
        if tok.startswith("-"):
            continue
        if _FILE_EXT_RE.search(tok) or "/" in tok:
            target = tok.rsplit("/", 1)[-1]
            break
    if target:
        return f"{base}:{target}"
    return base


def _normalize_to_set(output: str) -> frozenset:
    if not output:
        return frozenset()
    lines = output.strip().split("\n")[:MAX_OUTPUT_LINES]
    normalized = set()
    for line in lines:
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
        line = re.sub(r"pid[=: ]\d+", "pid=PID", line, flags=re.I)
        line = re.sub(r"/tmp/[^\s]+", "/tmp/TMPFILE", line)
        line = re.sub(r"\d+\.\d{3,}s", "N.NNNs", line)
        line = line.strip()
        if line:
            normalized.add(line)
    return frozenset(normalized)


def _jaccard(current_set: frozenset, previous_set: frozenset | None) -> float:
    if previous_set is None:
        return 0.0
    if not current_set and not previous_set:
        return 1.0
    union = current_set | previous_set
    return len(current_set & previous_set) / len(union) if union else 1.0


def _has_error_indicators(output: str) -> bool:
    if not output:
        return False
    return bool(ERROR_PATTERNS.search(output[:2000]))


def _strip_system_reminders(output: str) -> str:
    if not output or "<system-reminder" not in output:
        return output
    return _SYSTEM_REMINDER_RE.sub("", output)


def compute_step_features(steps: list[dict]) -> list[dict]:
    """Compute per-step features from normalized step dicts.

    Args:
        steps: list of normalized step dicts (tool, cmd, file, output, thinking)

    Returns:
        list of feature dicts with exactly STEP_FEATURES fields
    """
    if not steps:
        return []

    total_steps = len(steps)
    result = []
    output_history: dict = {}  # cmd_hash_int → output_set, for output_similarity

    for i, step in enumerate(steps):
        tool = step.get("tool", "other")
        if tool not in TOOL_TO_IDX:
            tool = "other"

        file_path = step.get("file")
        cmd = step.get("cmd", "")
        output = step.get("output", "")

        # Compute hashes as unsigned 32-bit integers for identity comparisons
        file_hash_int = zlib.crc32(file_path.encode()) & 0xFFFFFFFF if file_path else None
        if tool == "bash" and cmd:
            cmd_key = _cmd_semantic_key(cmd)
            cmd_hash_int = (
                zlib.crc32(cmd_key.encode()) & 0xFFFFFFFF if cmd_key else None
            )
        else:
            cmd_key = f"{tool}:{cmd}" if cmd else None
            cmd_hash_int = (
                zlib.crc32(cmd_key.encode()) & 0xFFFFFFFF if cmd_key else None
            )

        output = _strip_system_reminders(output)

        is_edit_tool = tool in ("edit", "create", "submit")
        if is_edit_tool:
            output_set = frozenset()
            has_prior = False
            output_sim = 0.0
        else:
            output_set = _normalize_to_set(output)
            has_prior = output_history.get(cmd_hash_int) is not None
            output_sim = _jaccard(output_set, output_history.get(cmd_hash_int))

        feat = {
            "tool_idx": TOOL_TO_IDX[tool],
            "cmd_hash": float(cmd_hash_int * _CRC32_NORM) if cmd_hash_int is not None else 0.0,
            "file_hash": float(file_hash_int * _CRC32_NORM) if file_hash_int is not None else 0.0,
            "output_similarity": float(output_sim),
            "has_prior_output": 1.0 if has_prior else 0.0,
            "output_length": float(math.log1p(output.count("\n"))),
            "is_error": 1.0 if _has_error_indicators(output) else 0.0,
            "step_index_norm": float(i) / float(max(total_steps - 1, 1)),
        }
        result.append(feat)

        if cmd_hash_int is not None and not is_edit_tool:
            output_history[cmd_hash_int] = output_set

    return result


def _is_valid_feature_file(path: str, n_steps: int) -> bool:
    """Check if a feature file is valid and matches expected step count."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("schema_version") == SCHEMA_VERSION
            and data.get("n_steps") == n_steps
            and len(data.get("steps", [])) == n_steps
        )
    except (json.JSONDecodeError, OSError):
        return False


def extract_session(
    steps: list[dict],
    session_id: str,
    source: str,
    out_dir: str,
    force: bool = False,
) -> str:
    """Extract features for one session. Returns path to feature file.

    Idempotent: skip if valid feature file exists (unless force=True).
    Validates: JSON validity, len(steps)==n_steps, schema_version match.

    Args:
        steps: list of normalized step dicts
        session_id: session identifier
        source: source name
        out_dir: output directory for feature files
        force: re-extract even if valid file exists

    Returns:
        path to the feature file
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session_id}_features.json")

    n_steps = len(steps)

    if not force and _is_valid_feature_file(out_path, n_steps):
        return out_path

    step_features = compute_step_features(steps)

    data = {
        "session_id": session_id,
        "source": source,
        "schema_version": SCHEMA_VERSION,
        "n_steps": n_steps,
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": step_features,
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=out_dir, delete=False, suffix=".tmp", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return out_path
