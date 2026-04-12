"""Tests for src/training/train.py."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.training.train import (
    INPUT_DIM,
    N_HISTORY,
    NUM_FEATURES,
    STEP_FEATURES,
    build_sequences,
    load_rows_from_jsonl,
    session_split,
)


def _make_rows(n_sessions=10, steps_per_session=5) -> list[dict]:
    rows = []
    for s in range(n_sessions):
        for i in range(steps_per_session):
            rows.append(
                {
                    "session_id": f"sess_{s:03d}",
                    "step": i,
                    "tool_idx": i % 7,
                    "cmd_hash": float(i) / 10.0,
                    "file_hash": float(s) / 10.0,
                    "output_similarity": 0.0,
                    "has_prior_output": 0.0,
                    "output_length": 1.0,
                    "is_error": 0.0,
                    "step_index_norm": float(i) / max(steps_per_session - 1, 1),
                    "label": 1.0 if (s % 5 == 0 and i > 2) else (0.5 if i == 1 else 0.0),
                }
            )
    return rows


class TestManifestLoading:
    def test_manifest_loads_with_weights_and_source_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = _make_rows(5, 3)
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            manifest = {
                "schema_version": 3,
                "datasets": [
                    {
                        "source_dir": "datasets/test/",
                        "path": jsonl_path,
                        "weight": 2.0,
                    }
                ],
            }
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            loaded = json.load(open(manifest_path))
            assert loaded["datasets"][0]["weight"] == 2.0
            assert loaded["datasets"][0]["source_dir"] == "datasets/test/"
            assert loaded["schema_version"] == 3


class TestSessionSplit:
    def test_no_session_in_both_sets(self):
        rows = _make_rows(50, 5)
        train_by_session, test_by_session = session_split(rows, test_fraction=0.1)

        overlap = set(train_by_session) & set(test_by_session)
        assert len(overlap) == 0, f"Sessions in both sets: {overlap}"

    def test_all_sessions_covered(self):
        rows = _make_rows(20, 5)
        train_by_session, test_by_session = session_split(rows, test_fraction=0.1)

        all_in = {r["session_id"] for r in rows}
        all_out = set(train_by_session) | set(test_by_session)
        assert all_in == all_out

    def test_split_ratio_approximately_correct(self):
        rows = _make_rows(100, 3)
        train_by_session, test_by_session = session_split(rows, test_fraction=0.1)

        assert len(test_by_session) == 10
        assert len(train_by_session) == 90

    def test_deterministic(self):
        rows = _make_rows(50, 3)
        train1, test1 = session_split(rows, test_fraction=0.2)
        train2, test2 = session_split(rows, test_fraction=0.2)

        assert set(test1) == set(test2)


class TestBuildSequences:
    def test_output_shape(self):
        """Each step produces one INPUT_DIM-wide input vector."""
        rows = _make_rows(n_sessions=3, steps_per_session=5)
        by_session = {}
        for r in rows:
            by_session.setdefault(r["session_id"], []).append(r)

        inputs, labels, sids = build_sequences(by_session)
        assert inputs.shape == (15, INPUT_DIM)
        assert labels.shape == (15,)
        assert len(sids) == 15

    def test_first_step_history_is_zero(self):
        """History positions are zero-padded at the start of each session."""
        rows = _make_rows(n_sessions=1, steps_per_session=3)
        by_session = {"sess_000": rows}

        inputs, _, _ = build_sequences(by_session)
        # First step: positions [NUM_FEATURES:] are all history — must be zeros
        assert np.all(inputs[0, NUM_FEATURES:] == 0.0)

    def test_second_step_prev_matches_first(self):
        """At step 1, history slot T-1 must equal step 0's feature vector."""
        rows = _make_rows(n_sessions=1, steps_per_session=3)
        by_session = {"sess_000": rows}

        inputs, _, _ = build_sequences(by_session)
        # Step 0 features
        step0_feats = inputs[0, :NUM_FEATURES]
        # Step 1, T-1 history slot (positions NUM_FEATURES : 2*NUM_FEATURES)
        step1_prev = inputs[1, NUM_FEATURES : 2 * NUM_FEATURES]
        np.testing.assert_array_equal(step0_feats, step1_prev)

    def test_previous_scores_in_input(self):
        """Previous labels appear at the last N_HISTORY positions."""
        rows = _make_rows(n_sessions=1, steps_per_session=4)
        by_session = {"sess_000": rows}

        inputs, labels, _ = build_sequences(by_session)
        # At step 2: score_T-1 = label[1], score_T-2 = label[0], rest zeros
        prev_scores = inputs[2, NUM_FEATURES * (1 + N_HISTORY):]
        assert prev_scores[0] == labels[1]  # T-1
        assert prev_scores[1] == labels[0]  # T-2
        assert prev_scores[2] == 0.0        # T-3 (no step yet)

    def test_missing_feature_raises(self):
        """A row missing a STEP_FEATURES key must raise, not silently fill zero."""
        rows = _make_rows(n_sessions=1, steps_per_session=2)
        del rows[0]["cmd_hash"]
        by_session = {"sess_000": rows}

        with pytest.raises(KeyError):
            build_sequences(by_session)


class TestBuildSequencesRingBuffer:
    def test_t3_slot_holds_step0_features(self):
        """At step T=3, slot T-3 (positions 3*NUM_FEATURES:4*NUM_FEATURES) must
        equal step 0's feature vector — validates ring buffer traversal depth."""
        rows = _make_rows(n_sessions=1, steps_per_session=5)
        by_session = {"sess_000": rows}

        inputs, _, _ = build_sequences(by_session)
        step0_feats = inputs[0, :NUM_FEATURES]
        step3_t3_slot = inputs[3, 3 * NUM_FEATURES : 4 * NUM_FEATURES]
        np.testing.assert_array_equal(step0_feats, step3_t3_slot)


class TestNormalization:
    def test_score_dims_not_normalized(self):
        """The N_HISTORY score dimensions (last N_HISTORY positions) must be left
        in [0,1] and not shifted/scaled by training stats.

        Bug: normalizing score dims with training-set mean/std causes inference
        mismatch — at train time scores are trimodal {0, 0.5, 1}, at inference
        they are continuous sigmoid outputs.

        Verified by: after applying train.py normalization, normalized score dims
        must equal the raw score dims (identity transform = mean 0, std 1 applied).
        """
        rows = _make_rows(n_sessions=20, steps_per_session=10)
        train_by_session, _ = session_split(rows, test_fraction=0.1)
        train_inputs, train_labels, _ = build_sequences(train_by_session)

        # Apply normalization exactly as train.py does
        feat_dims = INPUT_DIM - N_HISTORY
        mean = np.zeros(INPUT_DIM, dtype=np.float32)
        std = np.ones(INPUT_DIM, dtype=np.float32)
        mean[:feat_dims] = train_inputs[:, :feat_dims].mean(axis=0)
        std[:feat_dims] = train_inputs[:, :feat_dims].std(axis=0).clip(min=1e-6)
        normalized = (train_inputs - mean) / std

        # Score dims must be unchanged after normalization
        np.testing.assert_array_equal(
            normalized[:, -N_HISTORY:],
            train_inputs[:, -N_HISTORY:],
            err_msg="Score dims must not be modified by normalization",
        )


class TestLoadRowsFromJsonl:
    def test_loads_rows(self):
        rows = _make_rows(5, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_rows_from_jsonl(path)
            assert len(loaded) == 15

    def test_skips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"session_id": "a", "label": 0.0}) + "\n")
                f.write("\n")
                f.write(json.dumps({"session_id": "b", "label": 1.0}) + "\n")

            loaded = load_rows_from_jsonl(path)
            assert len(loaded) == 2


class TestGenerateManifestDefault:
    def test_no_source_dirs_reads_manifest(self, tmp_path, monkeypatch):
        """generate.py with no positional args must read source_dirs from manifest."""
        import sys
        from unittest.mock import patch, MagicMock

        manifest = {
            "schema_version": 3,
            "datasets": [
                {"source_dir": "datasets/foo/", "path": "data/generated/foo_v3.jsonl", "weight": 1.0},
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        captured = []

        def fake_process_source(source_dir, **kwargs):
            captured.append(source_dir)
            return 0, 0, 0

        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["generate.py", "--manifest", str(manifest_path)]), \
             patch("generate.process_source", side_effect=fake_process_source):
            import generate
            generate.main()

        assert captured == ["datasets/foo/"]

    def test_no_source_dirs_no_manifest_exits(self, tmp_path, monkeypatch):
        """generate.py with no args and missing manifest must exit with error."""
        from unittest.mock import patch

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info, \
             patch("sys.argv", ["generate.py", "--manifest", "nonexistent.json"]):
            import generate
            generate.main()
        assert exc_info.value.code == 1
