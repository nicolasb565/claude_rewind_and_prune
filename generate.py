"""Entry point: fetch → label → extract features → merge.

Usage:
  python generate.py [datasets/nlile/ ...] [--max-sessions N] [--force-relabel]
                     [--schema-version N] [--retry-failed] [--dry-run-estimate]
"""

import argparse
import gzip
import json
import os
import random
import sys
import tempfile
from fnmatch import fnmatch

# Ensure src is on the path
sys.path.insert(0, os.path.dirname(__file__))  # pylint: disable=wrong-import-position

from src.pipeline.batch_label import run_batch_label  # noqa: E402
from src.pipeline.extract_features import SCHEMA_VERSION, _is_valid_feature_file, extract_session  # noqa: E402
from src.pipeline.label_session import validate_label_file  # noqa: E402
from src.pipeline.merge_session import merge_session  # noqa: E402
from src.pipeline.migrate_features import migrate_artifact  # noqa: E402


def _load_config(source_dir: str) -> tuple[dict, dict]:
    fetch_path = os.path.join(source_dir, "fetch.json")
    filter_path = os.path.join(source_dir, "filter.json")
    with open(fetch_path, encoding="utf-8") as f:
        fetch_cfg = json.load(f)
    filter_cfg: dict = {}
    if os.path.exists(filter_path):
        with open(filter_path, encoding="utf-8") as f:
            filter_cfg = json.load(f)
    return fetch_cfg, filter_cfg


def _apply_filters(
    sessions: list[dict],
    filter_cfg: dict,
    source_name: str,  # pylint: disable=unused-argument
) -> list[dict]:
    min_steps = filter_cfg.get("min_steps", 0)
    max_steps = filter_cfg.get("max_steps", 999999)
    max_sessions = filter_cfg.get("max_sessions")
    folder_limits = filter_cfg.get("folder_limits", [])

    filtered = []
    for sess in sessions:
        n = len(sess.get("steps", []))
        if n < min_steps or n > max_steps:
            continue
        # folder_limits: check path field
        if folder_limits:
            path_val = sess.get("_source_path", "")
            skip = False
            for fl in folder_limits:
                pattern = fl.get("pattern", "")
                limit = fl.get("max", 0)
                if fnmatch(path_val, pattern):
                    if (
                        sum(
                            1
                            for s in filtered
                            if fnmatch(s.get("_source_path", ""), pattern)
                        )
                        >= limit
                    ):
                        skip = True
                        break
            if skip:
                continue
        filtered.append(sess)

    if max_sessions is not None and len(filtered) > max_sessions:
        rng = random.Random(42)
        session_ids = sorted(s["session_id"] for s in filtered)
        rng.shuffle(session_ids)
        keep = set(session_ids[:max_sessions])
        filtered = [s for s in filtered if s["session_id"] in keep]

    return filtered


def _fetch_parquet(fetch_cfg: dict, source_name: str) -> list[dict]:
    import pyarrow.parquet as pq  # pylint: disable=import-outside-toplevel
    from src.pipeline.parsers import (
        nlile as nlile_parser,
    )  # pylint: disable=import-outside-toplevel
    from src.pipeline.parsers import (
        dataclaw as dataclaw_parser,
    )  # pylint: disable=import-outside-toplevel

    parser_name = fetch_cfg.get("parser", "nlile")
    path = fetch_cfg["path"]

    sessions = []
    files = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))

    for fname in files:
        table = pq.read_table(os.path.join(path, fname))
        for i in range(len(table)):
            row_id = table.column("id")[i].as_py()
            msgs_raw = table.column("messages_json")[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)
            try:
                if parser_name == "nlile":
                    steps = nlile_parser.parse_session(msgs)
                else:
                    steps = dataclaw_parser.parse_session(msgs)
            except Exception:  # pylint: disable=broad-except
                continue
            sessions.append(
                {
                    "session_id": f"{source_name}_{row_id}",
                    "steps": steps,
                    "_source_path": fname,
                }
            )
    return sessions


def _fetch_huggingface(fetch_cfg: dict, source_name: str) -> list[dict]:
    import datasets as hf_datasets  # pylint: disable=no-name-in-module

    from src.pipeline.parsers import (
        dataclaw as dataclaw_parser,
    )  # pylint: disable=import-outside-toplevel
    from src.pipeline.parsers import (
        claudeset as claudeset_parser,
    )  # pylint: disable=import-outside-toplevel

    repo = fetch_cfg["repo"]
    split = fetch_cfg.get("split", "train")
    parser_name = fetch_cfg.get("parser", "dataclaw")
    model_filter = fetch_cfg.get("model_filter")

    dataset = hf_datasets.load_dataset(repo, split=split)
    sessions = []

    for row in dataset:
        # Model filter
        if model_filter and row.get("model") not in model_filter:
            continue

        session_id = row.get("session_id", row.get("id", ""))
        try:
            if parser_name == "claudeset":
                turns = row.get("turns", [])
                # Use parse_session (mixed) so compact blocks reach format_transcript
                # for labeling context. CompactBlocks are filtered when needed for
                # feature extraction and step counting (see process_source).
                steps = claudeset_parser.parse_session(turns)
            else:
                msgs = row.get("messages", [])
                steps = dataclaw_parser.parse_session(msgs)
        except Exception:  # pylint: disable=broad-except
            continue

        sessions.append(
            {
                "session_id": f"{source_name}_{session_id}",
                "steps": steps,
            }
        )

    return sessions


def _fetch_proprietary(fetch_cfg: dict, source_name: str) -> list[dict]:
    from src.pipeline.parsers import (
        dataclaw as dataclaw_parser,
    )  # pylint: disable=import-outside-toplevel
    from src.pipeline.parsers import (
        nlile as nlile_parser,
    )  # pylint: disable=import-outside-toplevel

    parser_name = fetch_cfg.get("parser", "dataclaw")
    path = fetch_cfg["path"]

    if not os.path.exists(path):
        print(f"  WARNING: proprietary path {path} not found, treating as labeled_gz")
        artifact = fetch_cfg.get("artifact")
        if artifact and os.path.exists(artifact):
            # Reject artifacts from the old window-label architecture (schema_version < 2).
            # Labels were per 10-step window; current pipeline requires per-step labels.
            # Regenerate from raw sessions using the current pipeline.
            try:
                with gzip.open(artifact, "rt") as _gz:
                    _first = json.loads(next(l for l in _gz if l.strip()))
                artifact_version = _first.get("schema_version", 1)
            except Exception:  # pylint: disable=broad-except
                artifact_version = 1
            if artifact_version < SCHEMA_VERSION:
                print(
                    f"  WARNING: artifact {artifact} is schema_version={artifact_version}, "
                    f"expected {SCHEMA_VERSION} (old window-label format — "
                    "regenerate from raw sessions using the current pipeline). Skipping."
                )
                return []
            return _read_labeled_gz(artifact, source_name)
        return []

    sessions = []
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        if fname.endswith(".jsonl"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    try:
                        if parser_name == "nlile":
                            steps = nlile_parser.parse_session(row.get("messages", []))
                        else:
                            steps = dataclaw_parser.parse_session(
                                row.get("messages", [])
                            )
                    except Exception:  # pylint: disable=broad-except
                        continue
                    session_id = row.get("session_id", row.get("id", fname))
                    sessions.append(
                        {
                            "session_id": f"{source_name}_{session_id}",
                            "steps": steps,
                            "_source_path": fpath,
                        }
                    )
    return sessions


def _read_labeled_gz(gz_path: str, source_name: str) -> list[dict]:
    """Read rows from a labeled .gz artifact (already merged JSONL).

    source_name is kept for API symmetry with other fetch functions.
    """
    _ = source_name
    sessions_by_id: dict[str, dict] = {}
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("session_id", "")
            if sid not in sessions_by_id:
                sessions_by_id[sid] = {"session_id": sid, "rows": []}
            sessions_by_id[sid]["rows"].append(row)
    return list(sessions_by_id.values())


def _update_gz_artifact(
    gz_path: str,
    new_rows: list[dict],
) -> None:
    """Append new rows to a .gz artifact (atomic write via temp-file-rename)."""
    existing: list[dict] = []
    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

    existing_ids = {r.get("session_id") for r in existing}
    to_add = [r for r in new_rows if r.get("session_id") not in existing_ids]

    if not to_add:
        return

    all_rows = existing + to_add
    dir_path = os.path.dirname(gz_path) or "."
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=dir_path, delete=False, suffix=".tmp.gz"
        ) as tmp:
            tmp_path = tmp.name
            with gzip.open(tmp, "wt") as gz_out:
                for row in all_rows:
                    gz_out.write(json.dumps(row) + "\n")
        os.replace(tmp_path, gz_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _load_progress(progress_path: str) -> dict:
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old progress files that lack done_sessions
        if "done_sessions" not in data:
            data["done_sessions"] = []
        return data
    return {
        "total": 0,
        "done": 0,
        "failed": 0,
        "pending": 0,
        "failed_sessions": [],
        "done_sessions": [],
    }


def _save_progress(progress_path: str, progress: dict) -> None:
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    tmp_path = None
    dir_path = os.path.dirname(progress_path)
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=dir_path, delete=False, suffix=".tmp", encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            json.dump(progress, tmp, indent=2)
        os.replace(tmp_path, progress_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def process_source(  # pylint: disable=too-many-branches,too-many-statements,too-many-positional-arguments
    source_dir: str,
    schema_version: int = SCHEMA_VERSION,
    max_sessions: int | None = None,
    force_relabel: bool = False,
    dry_run_estimate: bool = False,
    retry_failed: bool = False,
    skip_labeling: bool = False,
) -> tuple[int, int, int]:
    """Process one dataset source. Returns (done, failed, pending)."""
    source_name = os.path.basename(source_dir.rstrip("/"))
    fetch_cfg, filter_cfg = _load_config(source_dir)
    data_type = fetch_cfg.get("type", "parquet")

    labels_dir = os.path.join("data", "labels", source_name)
    features_dir = os.path.join("data", "features", source_name)
    generated_dir = "data/generated"
    out_jsonl = os.path.join(generated_dir, f"{source_name}_v{schema_version}.jsonl")
    progress_path = os.path.join(
        generated_dir, f"{source_name}_v{schema_version}_progress.json"
    )

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)

    progress = _load_progress(progress_path)

    # Always derive done_set from the actual JSONL content — this is the
    # authoritative record and also handles migration from old progress files
    # that predate the done_sessions field.
    done_set_from_jsonl: set[str] = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        _row = json.loads(_line)
                        done_set_from_jsonl.add(_row["session_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if done_set_from_jsonl:
        progress["done_sessions"] = sorted(done_set_from_jsonl)

    # Handle labeled_gz type
    if data_type == "labeled_gz":
        artifact = fetch_cfg.get("artifact") or fetch_cfg.get("path")
        if not artifact or not os.path.exists(artifact):
            print(f"  WARNING: labeled_gz artifact not found: {artifact}")
            return 0, 0, 0

        # Check schema version and migrate if needed
        migrate_artifact(artifact, to_version=schema_version)

        # Write directly to generated/
        session_rows: dict[str, list] = {}
        with gzip.open(artifact, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row.get("session_id", "")
                session_rows.setdefault(sid, []).append(row)

        done = 0
        with open(out_jsonl, "w", encoding="utf-8") as fout:
            for sid, rows in session_rows.items():
                for row in rows:
                    fout.write(json.dumps(row) + "\n")
                done += 1

        progress.update({"total": done, "done": done, "failed": 0, "pending": 0})
        _save_progress(progress_path, progress)
        return done, 0, 0

    # Fetch raw sessions
    print(f"  Fetching {source_name} (type={data_type})...")
    if data_type == "parquet":
        sessions = _fetch_parquet(fetch_cfg, source_name)
    elif data_type == "huggingface":
        sessions = _fetch_huggingface(fetch_cfg, source_name)
    elif data_type == "proprietary":
        artifact = fetch_cfg.get("artifact")
        if artifact and not os.path.exists(fetch_cfg["path"]):
            print("  WARNING: proprietary path not found, using labeled_gz fallback")
            if os.path.exists(artifact):
                sessions = _read_labeled_gz(artifact, source_name)
            else:
                sessions = []
        else:
            sessions = _fetch_proprietary(fetch_cfg, source_name)
    else:
        print(f"  WARNING: unknown type {data_type!r}")
        return 0, 0, 0

    # Apply filters
    sessions = _apply_filters(sessions, filter_cfg, source_name)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    print(f"  {len(sessions)} sessions after filtering")
    progress["total"] = len(sessions)

    # Retry failed
    if retry_failed and progress.get("failed_sessions"):
        retry_ids = {
            e["session_id"] if isinstance(e, dict) else e
            for e in progress["failed_sessions"]
        }
        sessions_retry = [s for s in sessions if s["session_id"] in retry_ids]
        sessions_new = [s for s in sessions if s["session_id"] not in retry_ids]
        sessions = sessions_retry + sessions_new
        progress["failed_sessions"] = []

    # Batch label
    if skip_labeling:
        results = {}
    else:
        results = run_batch_label(
            source_dir=source_dir,
            raw_sessions=sessions,
            labels_dir=labels_dir,
            max_sessions=max_sessions,
            dry_run_estimate=dry_run_estimate,
            force=force_relabel,
        )

    if dry_run_estimate:
        return 0, 0, len(sessions)

    # Report stale feature files that will be re-extracted due to schema bump
    stale = sum(
        1
        for sess in sessions
        if not _is_valid_feature_file(
            os.path.join(
                features_dir,
                f"{sess['session_id']}_features.json",
            ),
            sum(1 for s in sess.get("steps", []) if s.get("type") != "compact"),
        )
    )
    if stale:
        print(f"  Re-extracting features for {stale} sessions (schema v{schema_version})")

    # Extract features and merge
    done = 0
    failed = 0
    pending = 0
    artifact_rows: list[dict] = []
    done_set: set[str] = set(progress.get("done_sessions", []))

    for sess in sessions:
        sid = sess["session_id"]
        steps = sess.get("steps", [])
        # Filter out CompactBlocks (claudeset) — they carry context for the labeler
        # but are not real tool-call steps and must not enter feature extraction.
        real_steps = [s for s in steps if s.get("type") != "compact"]

        # Skip sessions already merged in a previous run
        if sid in done_set:
            done += 1
            continue

        labels = results.get(sid)

        if labels is None:
            label_path = os.path.join(labels_dir, f"{sid}_labels.json")
            if validate_label_file(label_path, len(real_steps)):
                with open(label_path, encoding="utf-8") as f:
                    labels = json.load(f)["labels"]
            else:
                pending += 1
                continue

        try:
            feat_path = extract_session(real_steps, sid, source_name, features_dir)
            label_path = os.path.join(labels_dir, f"{sid}_labels.json")
            merge_session(label_path, feat_path, out_jsonl)
            done += 1
            done_set.add(sid)

            # Build artifact row for proprietary sources
            if data_type == "proprietary":
                with open(feat_path, encoding="utf-8") as feat_f:
                    feat_steps = json.load(feat_f)["steps"]
                for i, (step_feats_row, label) in enumerate(zip(feat_steps, labels)):
                    row = {
                        "session_id": sid,
                        "step": i,
                        "schema_version": schema_version,
                    }
                    row.update(step_feats_row)
                    row["label"] = label
                    row["label_source"] = "batch_label"
                    artifact_rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ERROR processing {sid}: {exc}", file=sys.stderr)
            failed += 1
            progress["failed_sessions"].append({"session_id": sid, "error": str(exc)})

    progress.update(
        {
            "done": done,
            "failed": failed,
            "pending": pending,
            "done_sessions": sorted(done_set),
        }
    )
    _save_progress(progress_path, progress)

    # Update artifact for proprietary sources
    if data_type == "proprietary" and artifact_rows:
        artifact_path = fetch_cfg.get(
            "artifact", f"data/sources/{source_name}_labeled.jsonl.gz"
        )
        _update_gz_artifact(artifact_path, artifact_rows)

    return done, failed, pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dirs",
        nargs="*",
        default=[],
        help="Dataset directories to process (default: read from --manifest)",
    )
    parser.add_argument(
        "--manifest",
        default="training_manifest.json",
        help="Manifest file to read source_dirs from when none are given on the command line",
    )
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--force-relabel", action="store_true")
    parser.add_argument("--schema-version", type=int, default=SCHEMA_VERSION)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run-estimate", action="store_true")
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip batch labeling — only extract features and merge already-labeled sessions",
    )
    args = parser.parse_args()

    source_dirs = args.source_dirs
    if not source_dirs:
        if not os.path.exists(args.manifest):
            print(
                f"ERROR: no source dirs given and manifest not found: {args.manifest}",
                file=sys.stderr,
            )
            sys.exit(1)
        with open(args.manifest, encoding="utf-8") as _mf:
            _manifest = json.load(_mf)
        source_dirs = [
            entry["source_dir"]
            for entry in _manifest.get("datasets", [])
            if "source_dir" in entry
        ]
        if not source_dirs:
            print(
                f"ERROR: manifest {args.manifest} has no entries with 'source_dir'",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using source dirs from {args.manifest}: {source_dirs}")

    summary: dict[str, tuple[int, int, int]] = {}

    for source_dir in source_dirs:
        source_name = os.path.basename(source_dir.rstrip("/"))
        print(f"\nProcessing {source_name}...")
        try:
            done, failed, pending = process_source(
                source_dir,
                schema_version=args.schema_version,
                max_sessions=args.max_sessions,
                force_relabel=args.force_relabel,
                dry_run_estimate=args.dry_run_estimate,
                retry_failed=args.retry_failed,
                skip_labeling=args.skip_labeling,
            )
            summary[source_name] = (done, failed, pending)
        except Exception as exc:  # pylint: disable=broad-except
            if isinstance(exc, SystemExit):
                raise
            print(f"ERROR processing {source_name}: {exc}", file=sys.stderr)
            summary[source_name] = (0, 1, 0)

    print("\n=== Summary ===")
    total_done = total_failed = 0
    for name, (done, failed, pending) in summary.items():
        print(f"{name:25s} done={done}  failed={failed}  pending={pending}")
        total_done += done
        total_failed += failed

    print(f"\nTotal: {total_done} sessions done, {total_failed} failed")


if __name__ == "__main__":
    main()
