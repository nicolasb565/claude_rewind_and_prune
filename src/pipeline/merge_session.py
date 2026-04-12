"""Merge per-session label and feature files into JSONL training rows."""

import json
import os

from src.pipeline.extract_features import SCHEMA_VERSION

_LABEL_ENCODING = {
    "PRODUCTIVE": 0.0,
    "STUCK": 1.0,
    "UNSURE": 0.5,
}


def merge_session(
    labels_path: str,
    features_path: str,
    out_path: str,
) -> int:
    """Merge one session's labels and features into JSONL rows.

    Validates: n_steps match, schema_version current, label count matches.
    Appends rows to out_path (does not truncate).

    Args:
        labels_path: path to _labels.json file
        features_path: path to _features.json file
        out_path: path to output .jsonl file (appended to)

    Returns:
        number of rows written

    Raises:
        ValueError: if validation fails
    """
    with open(labels_path, encoding="utf-8") as f:
        label_data = json.load(f)

    with open(features_path, encoding="utf-8") as f:
        feature_data = json.load(f)

    # Validate schema version
    feat_version = feature_data.get("schema_version")
    if feat_version != SCHEMA_VERSION:
        raise ValueError(
            f"Feature file schema_version={feat_version!r}, expected {SCHEMA_VERSION}"
        )

    # Validate n_steps consistency
    label_n = label_data.get("n_steps")
    feat_n = feature_data.get("n_steps")
    labels = label_data.get("labels", [])
    steps = feature_data.get("steps", [])

    if label_n != feat_n:
        raise ValueError(
            f"n_steps mismatch: labels file has {label_n}, features file has {feat_n}"
        )
    if len(labels) != label_n:
        raise ValueError(f"Label count {len(labels)} != n_steps {label_n}")
    if len(steps) != feat_n:
        raise ValueError(f"Feature step count {len(steps)} != n_steps {feat_n}")

    session_id = feature_data.get("session_id", label_data.get("session_id", ""))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (feat, label_str) in enumerate(zip(steps, labels)):
            if label_str not in _LABEL_ENCODING:
                raise ValueError(
                    f"Unknown label string {label_str!r} at step {i} — "
                    f"expected one of {list(_LABEL_ENCODING)}"
                )
            label_val = _LABEL_ENCODING[label_str]
            row = {"session_id": session_id, "step": i}
            row.update(feat)
            row["label"] = label_val
            f.write(json.dumps(row) + "\n")
            rows_written += 1

    return rows_written
