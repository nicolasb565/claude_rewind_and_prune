#!/usr/bin/env python3
"""
Content / semantic features for stuck detection.

Targets the two failure modes identified by error analysis on 03_llvm:
  - FN: "agent hits the same error from different angles"
        → err_line_repeat_max_5, unique_err_sigs_6
  - FP: "agent running productive build/test iteration loop"
        → new_token_ratio_vs_5, has_success_marker, err_volume_ratio_vs_5

All features are pure text processing on step outputs — no LLM, no
embeddings. Designed to port cleanly to the JS proxy.

Pipeline:
  1. Build in-dist dataset from nlile parquet + labels
  2. Build OOD dataset from benchmarks/results/comparison_off transcripts
  3. Run correlation-flip validation for each new feature
  4. Train LR on 4-feature core + content survivors
  5. Evaluate on OOD benchmark head-to-head

Usage:
  .venv/bin/python benchmarks/v9_content_features.py --validate
  .venv/bin/python benchmarks/v9_content_features.py --train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import compute_step_features  # noqa: E402
from benchmarks.v9_tier1_train import compute_tier1_features  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
PARQUET_DIR = REPO / "data" / "separate" / "nlile_parquet" / "data"
LABEL_DIR = REPO / "data" / "labels" / "nlile"
OOD_DIR = REPO / "benchmarks" / "results" / "comparison_off"

# ── Constants for content features ─────────────────────────────────────────

# Error-line detector: matches lines that contain error-ish content.
# Union of extract_features.ERROR_PATTERNS plus file:line markers.
_ERR_LINE_RE = re.compile(
    r"(?i)(error|traceback|exception|failed|failure|fatal|cannot|unable to|"
    r"not found|permission denied|segmentation fault|core dumped|\bfail\b|"
    r"ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|"
    r"KeyError|AttributeError|RuntimeError|FileNotFoundError|undefined|"
    r"warning:|\berr:|[^\s]+\.[a-z]{1,5}:\d+[:.]|at \S+:\d+|panicked at)"
)

# Success marker — whole-word match (avoid matching "failed" accidentally).
_SUCCESS_RE = re.compile(
    r"(?i)(?<![a-z])("
    r"passed|succeeded|built|ok\b|done\b|completed|no errors?|all tests pass|"
    r"\d+ passed|\d+ tests? passed|compilation succeeded|finished successfully"
    r")(?![a-z])"
)
# Negations that should suppress the success match
_SUCCESS_NEG_RE = re.compile(
    r"(?i)(not ok|0 passed|failed|not passed|did not pass|failing|errors? found)"
)

# Output normalization (dedup hex/timestamps/tmpfile paths), mirrors
# extract_features._normalize_to_set but simpler.
_HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}")
_PID_RE = re.compile(r"pid[=: ]\d+", re.I)
_TMP_RE = re.compile(r"/tmp/[^\s]+")
_TIME_RE = re.compile(r"\d+\.\d{3,}s")
_NUM_RE = re.compile(r"\b\d{3,}\b")  # long integers get squashed
_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

# Error-core extractor: grab the semantic error message after any prefix.
# Matches things like: "error: undefined reference to 'foo'" → "undefined reference to 'foo'"
# Or "gcc: error: no such file or directory" → "no such file or directory"
_ERR_CORE_RE = re.compile(
    r"(?i)(?:error|exception|traceback|fatal|failed|failure)[:\s]+([^\n]+)"
)
_FILE_LINE_RE = re.compile(r"[\w./\-]+\.[a-zA-Z]{1,5}:\d+(?::\d+)?:?")

MAX_OUTPUT_CHARS = 8000  # only scan first 8k chars, same scale as v9


def _normalize_line(line: str) -> str:
    """Squash variable bits so the same error from different runs hashes same."""
    line = _HEX_RE.sub("0xADDR", line)
    line = _TS_RE.sub("TIMESTAMP", line)
    line = _PID_RE.sub("pid=PID", line)
    line = _TMP_RE.sub("/tmp/TMPFILE", line)
    line = _TIME_RE.sub("N.NNNs", line)
    line = _NUM_RE.sub("NUM", line)
    return line.strip()


def _err_line_hashes(output: str) -> set[int]:
    """Extract normalized-line fingerprints from the error-bearing lines of output."""
    if not output:
        return set()
    out = output[:MAX_OUTPUT_CHARS]
    hashes = set()
    for line in out.split("\n"):
        if _ERR_LINE_RE.search(line):
            norm = _normalize_line(line)
            if len(norm) >= 4:  # drop trivial
                hashes.add(zlib.crc32(norm.encode()) & 0xFFFFFFFF)
    return hashes


def _token_set(output: str) -> set[str]:
    if not output:
        return set()
    out = output[:MAX_OUTPUT_CHARS]
    return set(_WORD_RE.findall(out))


def _has_success(output: str) -> bool:
    if not output:
        return False
    out = output[:MAX_OUTPUT_CHARS]
    if _SUCCESS_NEG_RE.search(out):
        return False
    return bool(_SUCCESS_RE.search(out))


def _err_cores(output: str) -> set[int]:
    """Extract aggressively-normalized error message cores as hashes.

    For each line containing an error indicator, extract the text *after*
    the error keyword, strip file:line:col prefixes, normalize numbers/
    addresses. This lets 'gcc file.c:10: error: X' and 'clang other.c:42:
    error: X' collapse to the same signature if X is the same.
    """
    if not output:
        return set()
    out = output[:MAX_OUTPUT_CHARS]
    cores: set[int] = set()
    for line in out.split("\n"):
        if not _ERR_LINE_RE.search(line):
            continue
        # Strip file:line:col prefixes
        stripped = _FILE_LINE_RE.sub("", line)
        # Extract text after error keyword if present
        m = _ERR_CORE_RE.search(stripped)
        core = m.group(1) if m else stripped
        core = _normalize_line(core)
        # Additional aggressive normalization: lowercase, collapse whitespace
        core = re.sub(r"\s+", " ", core.lower()).strip()
        # Drop quoted identifiers (names of variables/functions specific to the file)
        core = re.sub(r"['\"`][^'\"`]{1,40}['\"`]", "NAME", core)
        if len(core) >= 8:
            cores.add(zlib.crc32(core.encode()) & 0xFFFFFFFF)
    return cores


def _err_line_count(output: str) -> int:
    if not output:
        return 0
    out = output[:MAX_OUTPUT_CHARS]
    return sum(1 for line in out.split("\n") if _ERR_LINE_RE.search(line))


# ── Per-step content feature computation over a session window ────────────

CONTENT_FEATURE_NAMES = [
    "err_line_repeat_max_5",
    "unique_err_sigs_6",
    "new_token_ratio_vs_5",
    "has_success_marker",
    "err_volume_ratio_vs_5",
    # Round 2 — targeted replacements based on distribution analysis
    "err_core_repeat_max_5",        # aggressive error-message normalization, same-core detection
    "output_content_jaccard_p1",    # avg jaccard of current output tokens vs p1
    "output_length_deriv",          # log-ratio of current output len vs p1 len
]


def compute_content_features(steps: list[dict]) -> list[dict]:
    """Annotate each step with content features. Steps must have 'output' field.

    Returns a list of dicts (one per step) containing only the content features.
    """
    err_hash_history: list[set[int]] = []  # last 5 prior: each is a set of fingerprints
    err_core_history: list[set[int]] = []  # last 5 prior: aggressive error cores
    token_history: list[set[str]] = []     # last 5 prior: each a token set
    err_count_history: list[int] = []      # last 5 prior: each an integer
    out_len_history: list[int] = []        # last 5 prior: each an integer

    out: list[dict] = []
    for step in steps:
        output = step.get("output", "") or ""
        cur_err_hashes = _err_line_hashes(output)
        cur_err_cores = _err_cores(output)
        cur_tokens = _token_set(output)
        cur_err_count = _err_line_count(output)
        cur_out_len = len(output)

        # Feature 1: err_line_repeat_max_5 — max count of any current error line
        # across the last 5 prior steps. If no current errors → 0.
        if not cur_err_hashes:
            err_line_repeat_max = 0.0
        else:
            # For each current hash, count how many of the 5 prior error sets contain it
            best_count = 0
            for h in cur_err_hashes:
                c = sum(1 for prior in err_hash_history if h in prior)
                if c > best_count:
                    best_count = c
            err_line_repeat_max = float(best_count) / 5.0

        # Feature 2: unique_err_sigs_6 — distinct error signatures across
        # current + up to 5 prior, normalized by 6 (saturates at 6).
        union = set(cur_err_hashes)
        for prior in err_hash_history:
            union |= prior
        unique_err_sigs = min(len(union), 6) / 6.0

        # Feature 3: new_token_ratio_vs_5 — fraction of current tokens not in
        # any prior. If current has no tokens → 0. If no history → 1 (all new).
        if not cur_tokens:
            new_token_ratio = 0.0
        elif not token_history:
            new_token_ratio = 1.0
        else:
            prior_union: set[str] = set()
            for t in token_history:
                prior_union |= t
            new_tokens = cur_tokens - prior_union
            new_token_ratio = len(new_tokens) / max(len(cur_tokens), 1)

        # Feature 4: has_success_marker
        has_success = 1.0 if _has_success(output) else 0.0

        # Feature 5: err_volume_ratio_vs_5 — log-ratio of current err count
        # vs the mean of the last 5 prior err counts. Centered at 0.
        if err_count_history:
            prior_mean = sum(err_count_history) / len(err_count_history)
            # log1p ratio, clamped to [-3, 3]
            ratio = math.log1p(cur_err_count) - math.log1p(prior_mean)
            ratio = max(-3.0, min(3.0, ratio))
        else:
            ratio = 0.0

        # Feature 6 (round 2): err_core_repeat_max_5 — aggressive-normalized
        # error cores. Detects "same semantic error from different file locations".
        if not cur_err_cores:
            err_core_repeat_max = 0.0
        else:
            best = 0
            for core_h in cur_err_cores:
                c = sum(1 for prior in err_core_history if core_h in prior)
                if c > best:
                    best = c
            err_core_repeat_max = float(best) / 5.0

        # Feature 7: output_content_jaccard_p1 — jaccard of current tokens vs p1
        if token_history and cur_tokens:
            p1_tokens = token_history[-1]
            if p1_tokens or cur_tokens:
                inter = len(cur_tokens & p1_tokens)
                union = len(cur_tokens | p1_tokens)
                jaccard_p1 = inter / max(union, 1)
            else:
                jaccard_p1 = 0.0
        else:
            jaccard_p1 = 0.0

        # Feature 8: output_length_deriv — log-ratio cur vs p1 length
        if out_len_history:
            p1_len = out_len_history[-1]
            out_len_deriv = math.log1p(cur_out_len) - math.log1p(p1_len)
            out_len_deriv = max(-4.0, min(4.0, out_len_deriv))
        else:
            out_len_deriv = 0.0

        out.append({
            "err_line_repeat_max_5": err_line_repeat_max,
            "unique_err_sigs_6": unique_err_sigs,
            "new_token_ratio_vs_5": new_token_ratio,
            "has_success_marker": has_success,
            "err_volume_ratio_vs_5": ratio,
            "err_core_repeat_max_5": err_core_repeat_max,
            "output_content_jaccard_p1": jaccard_p1,
            "output_length_deriv": out_len_deriv,
        })

        # Update histories AFTER recording
        err_hash_history.append(cur_err_hashes)
        if len(err_hash_history) > 5:
            err_hash_history.pop(0)
        err_core_history.append(cur_err_cores)
        if len(err_core_history) > 5:
            err_core_history.pop(0)
        token_history.append(cur_tokens)
        if len(token_history) > 5:
            token_history.pop(0)
        err_count_history.append(cur_err_count)
        if len(err_count_history) > 5:
            err_count_history.pop(0)
        out_len_history.append(cur_out_len)
        if len(out_len_history) > 5:
            out_len_history.pop(0)

    return out


# ── Dataset builders ───────────────────────────────────────────────────────

def iter_labeled_nlile_sessions(max_sessions: int | None = None) -> Iterable[tuple[str, list[dict], list[str]]]:
    """
    Yield (session_id, parsed_steps_with_output, labels) for nlile sessions
    that have both parquet rows and label files.
    """
    import pyarrow.parquet as pq

    # Index all label session ids first
    label_ids: set[str] = set()
    for f in os.listdir(LABEL_DIR):
        if f.endswith("_labels.json") and f.startswith("nlile_"):
            sid = f.replace("nlile_", "").replace("_labels.json", "")
            label_ids.add(sid)
    print(f"  {len(label_ids)} labeled nlile sessions available")

    yielded = 0
    for shard_idx in range(11):
        shard_path = PARQUET_DIR / f"train-{shard_idx:05d}-of-00011.parquet"
        if not shard_path.exists():
            continue
        tbl = pq.read_table(shard_path, columns=["id", "messages_json"]).to_pylist()
        for row in tbl:
            sid = row["id"]
            if sid not in label_ids:
                continue
            try:
                messages = json.loads(row["messages_json"])
                steps = parse_session(messages)
            except Exception:
                continue
            label_path = LABEL_DIR / f"nlile_{sid}_labels.json"
            try:
                labels = json.loads(label_path.read_text())["labels"]
            except Exception:
                continue
            n = min(len(steps), len(labels))
            if n == 0:
                continue
            yield sid, steps[:n], labels[:n]
            yielded += 1
            if max_sessions and yielded >= max_sessions:
                return


def build_indist_dataset(max_sessions: int | None = None) -> list[dict]:
    """
    Build in-dist rows. Each row has:
      - label: 1.0 STUCK / 0.0 PRODUCTIVE / 0.5 UNSURE
      - 4 core features + 5 content features (no slot features)
      - session_id, step
    """
    print(f"Building in-dist dataset (max_sessions={max_sessions})...")
    all_rows: list[dict] = []
    for sid, steps, labels in iter_labeled_nlile_sessions(max_sessions=max_sessions):
        v9_feats = compute_step_features(steps)
        compute_tier1_features(v9_feats)
        content = compute_content_features(steps)
        for i, (f, c, lbl) in enumerate(zip(v9_feats, content, labels)):
            row = {
                "session_id": sid,
                "step": i,
                "label": (1.0 if lbl == "STUCK" else
                          0.5 if lbl == "UNSURE" else 0.0),
                # Core features from earlier work
                "match_ratio_5": f["match_ratio_5"],
                "self_sim_max": f["self_sim_max"],
                "repeat_no_error": f["repeat_no_error"],
                "cur_bash_and_match_ratio": f["cur_bash_and_match_ratio"],
            }
            row.update(c)
            all_rows.append(row)
    print(f"  total rows: {len(all_rows)}")
    return all_rows


def build_ood_dataset() -> list[dict]:
    """Build OOD rows from benchmark transcripts."""
    print("Building OOD dataset from benchmark transcripts...")
    all_rows: list[dict] = []
    for td in sorted(OOD_DIR.iterdir()):
        if not td.is_dir():
            continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()):
            continue
        messages = []
        for line in t.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") in ("user", "assistant"):
                m = ev.get("message", {})
                if isinstance(m, dict):
                    messages.append(m)
        try:
            steps = parse_session(messages)
        except Exception:
            continue
        v9_feats = compute_step_features(steps)
        compute_tier1_features(v9_feats)
        content = compute_content_features(steps)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(v9_feats), len(labels))
        for i in range(n):
            f = v9_feats[i]
            c = content[i]
            lbl = labels[i]
            row = {
                "session_id": f"bench_{td.name}",
                "step": i,
                "label": (1.0 if lbl == "STUCK" else
                          0.5 if lbl == "UNSURE" else 0.0),
                "match_ratio_5": f["match_ratio_5"],
                "self_sim_max": f["self_sim_max"],
                "repeat_no_error": f["repeat_no_error"],
                "cur_bash_and_match_ratio": f["cur_bash_and_match_ratio"],
            }
            row.update(c)
            all_rows.append(row)
    print(f"  OOD rows: {len(all_rows)}")
    return all_rows


# ── Validation & training ──────────────────────────────────────────────────

def validate(indist: list[dict], ood: list[dict]):
    from scipy.stats import pointbiserialr

    def usable(rows):
        return [r for r in rows if r["label"] in (0.0, 1.0)]

    ind = usable(indist)
    ood = usable(ood)
    print(f"\nCorrelation-flip validation (in-dist n={len(ind)}, OOD n={len(ood)}):")
    print(f"{'feature':<28}{'ind_r':>10}{'ood_r':>10}{'verdict':>15}")
    print("-" * 63)
    for name in CONTENT_FEATURE_NAMES:
        X_i = np.array([r[name] for r in ind], dtype=np.float64)
        y_i = np.array([1 if r["label"] >= 0.9 else 0 for r in ind], dtype=np.int32)
        X_o = np.array([r[name] for r in ood], dtype=np.float64)
        y_o = np.array([1 if r["label"] >= 0.9 else 0 for r in ood], dtype=np.int32)
        r_i = pointbiserialr(y_i, X_i).statistic if X_i.std() > 1e-9 else 0.0
        r_o = pointbiserialr(y_o, X_o).statistic if X_o.std() > 1e-9 else 0.0
        i_s = "+" if r_i > 0.02 else ("-" if r_i < -0.02 else "≈0")
        o_s = "+" if r_o > 0.02 else ("-" if r_o < -0.02 else "≈0")
        if i_s == o_s and i_s != "≈0":
            v = "KEEP"
        elif (i_s == "+" and o_s == "-") or (i_s == "-" and o_s == "+"):
            v = "DROP (flip)"
        elif i_s == "≈0" and o_s == "≈0":
            v = "DROP (weak)"
        elif o_s == "≈0":
            v = "MARGINAL"
        else:
            v = "?"
        print(f"{name:<28}{r_i:>+10.4f}{r_o:>+10.4f}{v:>15}")


def train_lr(indist: list[dict], ood: list[dict], features: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def build(rows):
        X = np.array([[r[k] for k in features] for r in rows if r["label"] in (0.0, 1.0)], dtype=np.float64)
        y = np.array([1 if r["label"] >= 0.9 else 0 for r in rows if r["label"] in (0.0, 1.0)], dtype=np.int32)
        return X, y

    X_tr, y_tr = build(indist)
    X_ood, y_ood = build(ood)
    print(f"\nTraining LR on {len(features)} features, {len(X_tr)} rows")

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0).clip(min=1e-6)
    Xtr = (X_tr - mean) / std
    Xood = (X_ood - mean) / std

    print(f"{'mult':>6}{'ind_AUC':>10}{'ood_AUC':>10}"
          f"{'ood_P':>8}{'ood_R':>8}{'ood_F1':>8}{'TP':>5}{'FP':>5}{'FN':>5}")
    for mult in [1.0, 2.0, 3.0]:
        num_pos = int(y_tr.sum()); num_neg = len(y_tr) - num_pos
        pw = (num_neg / max(num_pos, 1)) * mult
        lr = LogisticRegression(
            C=1.0, class_weight={0: 1.0, 1: pw},
            max_iter=2000, solver="lbfgs",
        )
        lr.fit(Xtr, y_tr)
        s_tr = lr.predict_proba(Xtr)[:, 1]
        s_ood = lr.predict_proba(Xood)[:, 1]
        ind_auc = roc_auc_score(y_tr, s_tr)
        ood_auc = roc_auc_score(y_ood, s_ood)
        pred = (s_ood >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_ood == 1)).sum())
        fp = int(((pred == 1) & (y_ood == 0)).sum())
        fn = int(((pred == 0) & (y_ood == 1)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2*p*r/max(p+r, 1e-9)
        print(f"{mult:>6.1f}{ind_auc:>10.4f}{ood_auc:>10.4f}"
              f"{p:>8.3f}{r:>8.3f}{f1:>8.3f}{tp:>5}{fp:>5}{fn:>5}")

    # Final weights at mult=1
    num_pos = int(y_tr.sum()); num_neg = len(y_tr) - num_pos
    lr = LogisticRegression(
        C=1.0, class_weight={0: 1.0, 1: num_neg / max(num_pos, 1)},
        max_iter=2000, solver="lbfgs",
    )
    lr.fit(Xtr, y_tr)
    print(f"\nLR weights (mult=1, normalized):")
    print(f"  intercept                    {lr.intercept_[0]:+.4f}")
    for name, w in zip(features, lr.coef_[0]):
        print(f"  {name:<28} {w:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--max-sessions", type=int, default=1500,
                    help="cap on labeled in-dist sessions to process")
    ap.add_argument("--cache", default="data/generated/content_prototype.json",
                    help="cache file for prototype dataset")
    args = ap.parse_args()

    cache_path = REPO / args.cache
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}...")
        d = json.loads(cache_path.read_text())
        indist = d["indist"]; ood = d["ood"]
        print(f"  in-dist: {len(indist)}  OOD: {len(ood)}")
    else:
        indist = build_indist_dataset(max_sessions=args.max_sessions)
        ood = build_ood_dataset()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"indist": indist, "ood": ood}))
        print(f"Cached to {cache_path}")

    if args.validate:
        validate(indist, ood)

    if args.train:
        core = ["match_ratio_5", "self_sim_max", "repeat_no_error",
                "cur_bash_and_match_ratio"]
        top2 = ["unique_err_sigs_6", "new_token_ratio_vs_5"]
        top4 = top2 + ["has_success_marker", "err_volume_ratio_vs_5"]
        combos = [
            ("A: core4 baseline", core),
            ("B: core4 + top2 content (unique_err + new_tok)", core + top2),
            ("C: core4 + top4 content", core + top4),
            ("D: core4 + all 8 content", core + CONTENT_FEATURE_NAMES),
        ]
        for name, feats in combos:
            print(f"\n=== {name} ({len(feats)} features) ===")
            train_lr(indist, ood, feats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
