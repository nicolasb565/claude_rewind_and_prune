"""Extract per-step features from parsed session steps.

Feature computation logic is ported from abstract_trajectory.py and kept
self-contained here so the new pipeline has no dependency on the old src/.

Schema 6 (v9): adds 34 RELATIONAL features alongside the legacy schema-3
v5 features. The v9 features are intended to be used on their own (the
rest of the columns are kept for backward compat and for A/B experiments).

v9 layout per step: 5 prior-step features (6 each, history depth 5) + 4
current-step features = 34 floats. Each prior-slot feature is computed by
comparing that prior step to the CURRENT step — match on (action,
target_file, target_scope), self-relative output_similarity, output_length,
is_error. Current-step features are output_length, is_error,
output_similarity_vs_match, consecutive_match_count.

See benchmarks/v9_experiment.py for the design justification.
"""

import json
import math
import os
import re
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = 6

# Legacy v5 features (schema 3), kept for backward compatibility. Do not remove.
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

# v9 relational features (schema 6). A separate flat vector per step — the
# history is already embedded in the prior-slot features, so training on
# these does NOT need a ring buffer over them.
V9_N_HISTORY = 5
V9_SCOPE_DEPTH = 5
V9_FEATURE_NAMES = (
    [f"v9_p{i+1}_{k}"
     for i in range(V9_N_HISTORY)
     for k in ("act_match", "file_match", "scope_match", "self_sim", "out_len", "is_err")]
    + ["v9_cur_out_len", "v9_cur_is_err", "v9_cur_sim_vs_match", "v9_cur_consec_match"]
)
assert len(V9_FEATURE_NAMES) == 6 * V9_N_HISTORY + 4  # 34

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

# v9 helpers
_V9_PATH_TOKEN_RE = re.compile(
    r"(?:/?[\w@.\-]+/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}"
)
_V9_SUBCOMMAND_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\-]*$")
_V9_PROGS_WITH_INLINE_SCRIPT = {
    "node", "python", "python3", "ruby", "perl",
    "sh", "bash", "zsh", "fish", "awk", "sed", "tclsh",
}
_V9_INLINE_SCRIPT_FLAGS = {"-e", "-c", "--command", "--eval", "-p", "-P"}

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


# ─── v9 relational-feature helpers ────────────────────────────────────────

def _v9_action_of(step: dict) -> tuple:
    """See benchmarks/v9_experiment.py for the rationale. Summary:
      - non-bash: (tool_category, tool_name) — all Greps match each other
      - bash with inline-script flag (node -e / python -c): ('bash', prog)
      - bash: ('bash', prog, subcommand?) where subcommand is the immediate
        next token if it's an identifier and no flag has come before
    """
    tool = step.get("tool", "other")
    cmd = step.get("cmd", "") or ""
    if tool != "bash":
        return (tool, step.get("tool_name") or tool)

    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not _SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        return ("bash", tokens[0] if tokens else "")

    first_pipe = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first_pipe.strip().split()
    if not tokens:
        return ("bash", "")

    prog = tokens[0].rsplit("/", 1)[-1]
    if prog in _V9_PROGS_WITH_INLINE_SCRIPT:
        if any(t in _V9_INLINE_SCRIPT_FLAGS for t in tokens[1:]):
            return ("bash", prog)

    if len(tokens) >= 2:
        tok = tokens[1]
        if (not tok.startswith("-")
                and "/" not in tok
                and "." not in tok
                and _V9_SUBCOMMAND_RE.match(tok)):
            return ("bash", prog, tok)
    return ("bash", prog)


def _v9_target_file_of(step: dict) -> str | None:
    if step.get("file"):
        return str(step["file"])
    tool = step.get("tool", "other")
    cmd = step.get("cmd", "") or ""
    if tool == "bash":
        m = _V9_PATH_TOKEN_RE.search(cmd)
        return m.group(0) if m else None
    return cmd or None


def _v9_target_scope_of(step: dict) -> str | None:
    tf = _v9_target_file_of(step)
    if not tf or "/" not in tf:
        return None
    parts = tf.split("/")
    if len(parts) <= V9_SCOPE_DEPTH:
        return "/".join(parts[:-1]) if len(parts) > 1 else None
    return "/".join(parts[:V9_SCOPE_DEPTH])


@dataclass
class _V9StepInfo:
    action: tuple
    target_file: str | None
    target_scope: str | None
    output_set: frozenset
    output_length: float
    is_error: float
    self_relative_sim: float


def compute_v9_features(steps: list[dict]) -> list[list[float]]:
    """Produce 34-dim v9 feature vectors, one per step. Pure function."""
    history_by_match_key: dict[tuple, list[frozenset]] = {}
    infos: list[_V9StepInfo] = []
    for step in steps:
        action = _v9_action_of(step)
        target_file = _v9_target_file_of(step)
        target_scope = _v9_target_scope_of(step)
        clean = _strip_system_reminders(step.get("output", "") or "")
        output_set = _normalize_to_set(clean)
        match_key = (action, target_file)
        priors = history_by_match_key.get(match_key, [])
        self_sim = 0.0
        for p in priors:
            j = _jaccard(output_set, p)
            if j > self_sim:
                self_sim = j
                if self_sim >= 1.0:
                    break
        infos.append(_V9StepInfo(
            action=action,
            target_file=target_file,
            target_scope=target_scope,
            output_set=output_set,
            output_length=float(math.log1p(clean.count("\n"))),
            is_error=1.0 if _has_error_indicators(clean) else 0.0,
            self_relative_sim=float(self_sim),
        ))
        slots = history_by_match_key.setdefault(match_key, [])
        slots.append(output_set)
        if len(slots) > 5:
            slots.pop(0)

    result: list[list[float]] = []
    for t, cur in enumerate(infos):
        vec: list[float] = []
        prior_act_match = [0.0] * V9_N_HISTORY
        prior_file_match = [0.0] * V9_N_HISTORY
        for slot in range(V9_N_HISTORY):
            idx = t - 1 - slot
            if idx < 0:
                vec.extend([0.0] * 6)
                continue
            prior = infos[idx]
            am = 1.0 if prior.action == cur.action else 0.0
            fm = 1.0 if (prior.target_file is not None
                         and prior.target_file == cur.target_file) else 0.0
            sm = 1.0 if (prior.target_scope is not None
                         and prior.target_scope == cur.target_scope) else 0.0
            prior_act_match[slot] = am
            prior_file_match[slot] = fm
            vec.extend([am, fm, sm,
                        prior.self_relative_sim,
                        prior.output_length,
                        prior.is_error])
        vec.append(cur.output_length)
        vec.append(cur.is_error)
        out_sim_match = 0.0
        for slot in range(V9_N_HISTORY):
            idx = t - 1 - slot
            if idx < 0:
                break
            prior = infos[idx]
            if (prior.action == cur.action
                    and prior.target_file == cur.target_file
                    and prior.target_file is not None):
                out_sim_match = _jaccard(cur.output_set, prior.output_set)
                break
        vec.append(float(out_sim_match))
        matches = sum(
            1 for slot in range(V9_N_HISTORY)
            if prior_act_match[slot] == 1.0 and prior_file_match[slot] == 1.0
        )
        vec.append(matches / V9_N_HISTORY)
        result.append(vec)
    return result


def compute_step_features(steps: list[dict]) -> list[dict]:
    """Compute per-step features from normalized step dicts.

    Args:
        steps: list of normalized step dicts (tool, cmd, file, output, thinking)

    Returns:
        list of feature dicts with all STEP_FEATURES fields plus all
        V9_FEATURE_NAMES fields (schema 6). Each step dict has 8 legacy
        v5 features + 34 v9 features = 42 fields.
    """
    if not steps:
        return []
    # v9 features computed upfront so we can zip them into the v5 loop below
    v9_vectors = compute_v9_features(steps)

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
        # Merge v9 features (schema 6). Column order matches V9_FEATURE_NAMES.
        for name, value in zip(V9_FEATURE_NAMES, v9_vectors[i]):
            feat[name] = float(value)
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
