"""Tests for src/pipeline/batch_label.py — all API calls mocked."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from src.pipeline.label_session import write_label_file


def _make_transcripts(n=3):
    return [(f"sess_{i}", f"transcript {i}", 5) for i in range(n)]


def _make_mock_client():
    client = MagicMock()
    return client


class TestParseCsvToLabels:
    """CSV mapping → PRODUCTIVE/STUCK/UNSURE."""

    def test_csv_maps_correctly(self):
        from src.pipeline.label_session import parse_csv_labels

        assert parse_csv_labels("P,S,U", 3) == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_all_productive(self):
        from src.pipeline.label_session import parse_csv_labels

        assert parse_csv_labels("P,P,P,P,P", 5) == ["PRODUCTIVE"] * 5


class TestSubmitBatch:
    def test_submit_saves_pending_batch(self):
        mock_batch = MagicMock()
        mock_batch.id = "batch_abc123"

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_client.messages.batches.create.return_value = mock_batch
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import submit_batch

                transcripts = _make_transcripts(2)
                batch_id = submit_batch(transcripts, "nlile", tmpdir)

                assert batch_id == "batch_abc123"
                pending_path = os.path.join(tmpdir, "pending_batch.json")
                assert os.path.exists(pending_path)
                with open(pending_path) as f:
                    data = json.load(f)
                assert data["batch_id"] == "batch_abc123"

    def test_submit_saves_session_n_steps_in_pending(self):
        """pending_batch.json must record session_n_steps so resume works correctly
        even if the caller's session list changes between submission and resume."""
        mock_batch = MagicMock()
        mock_batch.id = "batch_nsteps"

        transcripts = [("sess_A", "transcript A", 10), ("sess_B", "transcript B", 25)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_client.messages.batches.create.return_value = mock_batch
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import submit_batch

                submit_batch(transcripts, "nlile", tmpdir)

                pending_path = os.path.join(tmpdir, "pending_batch.json")
                with open(pending_path) as f:
                    data = json.load(f)

                assert data["session_n_steps"] == {"sess_A": 10, "sess_B": 25}


class TestResumeFromPending:
    def test_resume_uses_saved_n_steps_not_current_list(self):
        """On resume, n_steps must come from pending_batch.json (what was submitted),
        not from the current session list (which may have changed).

        Bug: before the fix, if the session list changed between submission and resume,
        poll_and_retrieve received wrong n_steps and would fail label validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # pending_batch.json records sess_0 with n_steps=20
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump(
                    {
                        "batch_id": "resume_batch",
                        "source": "nlile",
                        "session_n_steps": {"sess_0": 20},
                    },
                    f,
                )

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()
                mock_status = MagicMock()
                mock_status.processing_status = "ended"
                mock_client.messages.batches.retrieve.return_value = mock_status

                # Batch returns 20 labels for sess_0
                mock_result = MagicMock()
                mock_result.custom_id = "sess_0"
                mock_result.result.type = "succeeded"
                mock_block = MagicMock()
                mock_block.text = ",".join(["P"] * 20)
                mock_result.result.message.content = [mock_block]
                mock_client.messages.batches.results.return_value = [mock_result]
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import run_batch_label

                # Current session list has sess_0 with n_steps=5 (DIFFERENT from submitted)
                raw_sessions = [{"session_id": "sess_0", "steps": [{}] * 5}]
                result = run_batch_label(
                    source_dir="datasets/nlile/",
                    raw_sessions=raw_sessions,
                    labels_dir=tmpdir,
                )

                # Labels should be written using saved n_steps=20, not current n_steps=5.
                # With the fix: label file written with 20 labels → result is not None.
                # Without the fix: n_steps=5 would cause parse_csv_labels to fail
                # (20 labels for 5 steps) → session left unlabeled.
                label_path = os.path.join(tmpdir, "sess_0_labels.json")
                assert os.path.exists(
                    label_path
                ), "label file should be written on resume"
                with open(label_path) as f:
                    label_data = json.load(f)
                assert label_data["n_steps"] == 20

    def test_pending_batch_skips_submission(self):
        """If pending_batch.json exists, poll without resubmitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a pending batch file
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump({"batch_id": "existing_batch_id", "source": "nlile"}, f)

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()

                # Batch is already ended
                mock_batch_status = MagicMock()
                mock_batch_status.processing_status = "ended"
                mock_client.messages.batches.retrieve.return_value = mock_batch_status
                mock_client.messages.batches.results.return_value = []
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import run_batch_label

                # Sessions that are already labeled — so nothing to submit
                with tempfile.TemporaryDirectory() as tmpdir2:
                    # Pre-write labels for all sessions
                    for i in range(2):
                        sid = f"sess_{i}"
                        write_label_file(
                            os.path.join(tmpdir, f"{sid}_labels.json"),
                            sid,
                            "nlile",
                            ["PRODUCTIVE"] * 5,
                            5,
                        )

                    raw_sessions = [
                        {"session_id": f"sess_{i}", "steps": [{}] * 5} for i in range(2)
                    ]
                    result = run_batch_label(
                        source_dir="datasets/nlile/",
                        raw_sessions=raw_sessions,
                        labels_dir=tmpdir,
                    )

                # Should not have called create (all already labeled)
                mock_client.messages.batches.create.assert_not_called()


class TestBatchExpiry:
    def test_expired_batch_deletes_pending_and_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pending_path = os.path.join(tmpdir, "pending_batch.json")
            with open(pending_path, "w") as f:
                json.dump({"batch_id": "expiring_batch", "source": "nlile"}, f)

            with patch("src.pipeline.batch_label._get_client") as mock_get_client:
                mock_client = _make_mock_client()

                mock_batch_status = MagicMock()
                mock_batch_status.processing_status = "expired"
                mock_client.messages.batches.retrieve.return_value = mock_batch_status
                mock_client.messages.batches.results.return_value = []
                mock_get_client.return_value = mock_client

                from src.pipeline.batch_label import poll_and_retrieve

                transcripts_by_id = {"sess_0": ("transcript", 5)}
                results = poll_and_retrieve(
                    "expiring_batch", "nlile", tmpdir, transcripts_by_id
                )

                # pending_batch.json should be deleted after expiry
                assert not os.path.exists(pending_path)
                # Session result should be None (not labeled)
                assert results.get("sess_0") is None


class TestRetryBackoff:
    def test_529_retries_with_backoff(self):
        """HTTP 529 triggers up to 4 retries with exponential backoff."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            # Simulate overloaded error then success
            error_429 = Exception("overloaded")
            error_429.status_code = 529

            success_batch = MagicMock()
            success_batch.id = "batch_ok"
            mock_client.messages.batches.create.side_effect = [
                error_429,
                success_batch,
            ]
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    batch_id = submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert batch_id == "batch_ok"
                    assert mock_client.messages.batches.create.call_count == 2

    def test_400_no_retry_exits(self):
        """HTTP 400 aborts immediately without retry."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            error_400 = Exception("bad request")
            error_400.status_code = 400
            mock_client.messages.batches.create.side_effect = error_400
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(SystemExit) as exc_info:
                        submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert exc_info.value.code == 1
                    # Should not retry at all
                    assert mock_client.messages.batches.create.call_count == 1

    def test_401_no_retry_exits(self):
        """HTTP 401 aborts immediately."""
        with patch("src.pipeline.batch_label._get_client") as mock_get_client:
            mock_client = _make_mock_client()

            error_401 = Exception("unauthorized")
            error_401.status_code = 401
            mock_client.messages.batches.create.side_effect = error_401
            mock_get_client.return_value = mock_client

            with patch("src.pipeline.batch_label.time.sleep"):
                from src.pipeline.batch_label import submit_batch

                with tempfile.TemporaryDirectory() as tmpdir:
                    with pytest.raises(SystemExit) as exc_info:
                        submit_batch(_make_transcripts(1), "nlile", tmpdir)
                    assert exc_info.value.code == 1


class TestCollectBatchResults:
    def _make_success(self, session_id: str, csv: str):
        from types import SimpleNamespace

        block = SimpleNamespace(text=csv)
        message = SimpleNamespace(content=[block])
        result_inner = SimpleNamespace(type="succeeded", message=message)
        return SimpleNamespace(custom_id=session_id, result=result_inner)

    def _make_error(self, session_id: str, err_type: str):
        from types import SimpleNamespace

        error = SimpleNamespace(type=err_type)
        result_inner = SimpleNamespace(type="errored", error=error)
        return SimpleNamespace(custom_id=session_id, result=result_inner)

    def test_good_response_written_and_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_1": ("transcript", 3)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_success("sess_1", "P,S,U"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            results, failures = _collect_batch_results(
                client, "batch_x", "test_src", tmpdir, transcripts
            )
            assert results["sess_1"] == ["PRODUCTIVE", "STUCK", "UNSURE"]
            assert failures == []
            assert os.path.exists(os.path.join(tmpdir, "sess_1_labels.json"))

    def test_bad_csv_count_goes_to_failures(self):
        """4 labels for a 3-step session must land in parse_failure_ids, not be dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_bad": ("transcript", 3)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_success("sess_bad", "P,S,U,P"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            results, failures = _collect_batch_results(
                client, "batch_x", "test_src", tmpdir, transcripts
            )
            assert results["sess_bad"] is None
            assert "sess_bad" in failures

    def test_off_by_one_still_fails(self):
        """±1 mismatch must trigger retry, not silent accept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts_short = {"sess_short": ("t", 3)}
            transcripts_long = {"sess_long": ("t", 3)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_success("sess_short", "P,S"),  # 2 for expected 3
            ]
            from src.pipeline.batch_label import _collect_batch_results

            _, failures = _collect_batch_results(
                client, "batch_x", "test_src", tmpdir, transcripts_short
            )
            assert "sess_short" in failures

    def test_unknown_session_id_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_known": ("transcript", 2)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_success("sess_known", "P,S"),
                self._make_success("sess_unknown", "P"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            results, _ = _collect_batch_results(
                client, "batch_x", "test_src", tmpdir, transcripts
            )
            assert "sess_unknown" not in results

    def test_non_recoverable_error_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_1": ("t", 2)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_error("sess_1", "billing_error"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            with pytest.raises(SystemExit) as exc_info:
                _collect_batch_results(
                    client, "batch_x", "test_src", tmpdir, transcripts
                )
            assert exc_info.value.code == 1

    def test_missing_session_warns(self, capsys):
        """Session in transcripts_by_id but absent from API results must log a warning.

        Issue #4: currently returns None silently — callers can't distinguish
        'API dropped it' from 'API returned an error for it'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_present": ("t", 2), "sess_missing": ("t", 3)}
            client = MagicMock()
            # API only returns results for sess_present
            client.messages.batches.results.return_value = [
                self._make_success("sess_present", "P,S"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            results, _ = _collect_batch_results(
                client, "batch_x", "test_src", tmpdir, transcripts
            )
            assert results["sess_missing"] is None
            captured = capsys.readouterr()
            assert "sess_missing" in captured.err

    def test_errored_session_not_warned_as_missing(self, capsys):
        """An errored session must NOT produce 'missing from batch results' warning.

        The API returned it — as an error — so it is not missing.
        Bug: the old implementation checked 'labels is None and not in parse_failure_ids',
        which is also true for errored sessions, producing a false-positive warning.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_errored": ("t", 2)}
            client = MagicMock()
            client.messages.batches.results.return_value = [
                self._make_error("sess_errored", "overloaded_error"),
            ]
            from src.pipeline.batch_label import _collect_batch_results

            _collect_batch_results(client, "batch_x", "test_src", tmpdir, transcripts)
            captured = capsys.readouterr()
            assert "missing from batch results" not in captured.err


class TestRetryParseFailures:
    def _make_success(self, session_id: str, csv: str):
        from types import SimpleNamespace

        block = SimpleNamespace(text=csv)
        message = SimpleNamespace(content=[block])
        result_inner = SimpleNamespace(type="succeeded", message=message)
        return SimpleNamespace(custom_id=session_id, result=result_inner)

    def _ended_status(self):
        from types import SimpleNamespace

        return SimpleNamespace(processing_status="ended")

    def test_retry_success_returns_labels(self):
        """Session that failed first parse succeeds on retry — labels are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_retry": ("transcript", 2)}
            client = MagicMock()
            # retrieve must return "ended" so _poll_until_done exits immediately
            client.messages.batches.retrieve.return_value = self._ended_status()
            client.messages.batches.results.return_value = [
                self._make_success("sess_retry", "P,S"),
            ]

            with patch(
                "src.pipeline.batch_label.submit_batch", return_value="batch_retry"
            ), patch("src.pipeline.batch_label._get_client", return_value=client):
                from src.pipeline.batch_label import _retry_parse_failures

                results = _retry_parse_failures(
                    client, "test_src", tmpdir, transcripts, ["sess_retry"]
                )
            assert results["sess_retry"] == ["PRODUCTIVE", "STUCK"]

    def test_retry_still_failing_discards_and_warns(self, capsys):
        """Session that fails parse again after retry is discarded with a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts = {"sess_bad": ("transcript", 2)}
            client = MagicMock()
            client.messages.batches.retrieve.return_value = self._ended_status()
            # Retry also returns wrong count (3 for expected 2)
            client.messages.batches.results.return_value = [
                self._make_success("sess_bad", "P,S,U"),
            ]

            with patch(
                "src.pipeline.batch_label.submit_batch", return_value="batch_retry"
            ), patch("src.pipeline.batch_label._get_client", return_value=client):
                from src.pipeline.batch_label import _retry_parse_failures

                results = _retry_parse_failures(
                    client, "test_src", tmpdir, transcripts, ["sess_bad"]
                )
            assert results["sess_bad"] is None
            captured = capsys.readouterr()
            assert "discarding" in captured.err
            assert "sess_bad" in captured.err

    def test_empty_failure_ids_does_not_submit(self):
        """_retry_parse_failures with empty list must not call submit_batch.

        Issue #9: currently would submit an empty batch, which the API rejects.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.pipeline.batch_label.submit_batch") as mock_submit:
                from src.pipeline.batch_label import _retry_parse_failures

                results = _retry_parse_failures(MagicMock(), "test_src", tmpdir, {}, [])
            mock_submit.assert_not_called()
            assert results == {}

    def test_empty_transcript_skipped_with_warning(self, capsys):
        """Session with empty transcript in failure_ids must be skipped, not retried.

        Issue #7: on resume, sessions filtered from the current list get
        empty-string transcripts. Retrying them sends '' to the API, guaranteed
        to produce 0 labels for n_steps > 0 — causing another discard loop.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # sess_empty has empty transcript (simulates resume scenario)
            transcripts = {"sess_empty": ("", 5), "sess_ok": ("real transcript", 2)}
            client = MagicMock()
            client.messages.batches.retrieve.return_value = self._ended_status()
            client.messages.batches.results.return_value = [
                self._make_success("sess_ok", "P,S"),
            ]

            with patch(
                "src.pipeline.batch_label.submit_batch", return_value="batch_retry"
            ) as mock_submit, patch(
                "src.pipeline.batch_label._get_client", return_value=client
            ):
                from src.pipeline.batch_label import _retry_parse_failures

                results = _retry_parse_failures(
                    client,
                    "test_src",
                    tmpdir,
                    transcripts,
                    ["sess_empty", "sess_ok"],
                )
            captured = capsys.readouterr()
            # sess_empty must be warned about and skipped
            assert "sess_empty" in captured.err
            assert results.get("sess_empty") is None
            # sess_ok still gets labeled
            assert results.get("sess_ok") == ["PRODUCTIVE", "STUCK"]
            # sess_empty must not appear in what was submitted to the API
            submitted_transcripts = mock_submit.call_args.args[0]
            submitted_ids = [sid for sid, _, _ in submitted_transcripts]
            assert "sess_empty" not in submitted_ids
            assert "sess_ok" in submitted_ids


class TestPollAndRetrieveRetry:
    """End-to-end retry path through poll_and_retrieve — issues #1, #2, #10."""

    def _make_success(self, session_id: str, csv: str):
        from types import SimpleNamespace

        block = SimpleNamespace(text=csv)
        message = SimpleNamespace(content=[block])
        result_inner = SimpleNamespace(type="succeeded", message=message)
        return SimpleNamespace(custom_id=session_id, result=result_inner)

    def _ended_status(self):
        from types import SimpleNamespace

        return SimpleNamespace(processing_status="ended")

    def test_retry_poll_precedes_retry_results(self):
        """poll must complete for the retry batch before its results are read.

        Issue #1: _retry_parse_failures was calling _collect_batch_results immediately
        after submit_batch. Fix: _poll_until_done added between submit and collect.

        This test asserts strict ordering: retrieve(retry_batch_id) must appear
        in mock_calls before the second results() call.
        """
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmpdir:
            call_log: list[tuple[str, str]] = []

            client = MagicMock()

            def tracking_retrieve(batch_id):
                call_log.append(("retrieve", batch_id))
                return self._ended_status()

            raw_results = [
                [self._make_success("sess_1", "P,S,U,EXTRA")],  # main: wrong count
                [self._make_success("sess_1", "P,S,U")],  # retry: correct
            ]
            results_seq = iter(raw_results)

            def tracking_results(batch_id):
                call_log.append(("results", batch_id))
                return iter(next(results_seq))

            client.messages.batches.retrieve.side_effect = tracking_retrieve
            client.messages.batches.results.side_effect = tracking_results
            client.messages.batches.create.return_value = SimpleNamespace(
                id="retry_batch_id"
            )

            with patch("src.pipeline.batch_label._get_client", return_value=client):
                from src.pipeline.batch_label import poll_and_retrieve

                results = poll_and_retrieve(
                    "main_batch",
                    "nlile",
                    tmpdir,
                    {"sess_1": ("transcript", 3)},
                )

        # retrieve("retry_batch_id") must appear before the second results call
        retrieve_retry_pos = next(
            i
            for i, (op, bid) in enumerate(call_log)
            if op == "retrieve" and bid == "retry_batch_id"
        )
        results_calls = [i for i, (op, _) in enumerate(call_log) if op == "results"]
        assert len(results_calls) == 2, f"expected 2 results calls, got: {call_log}"
        retry_results_pos = results_calls[1]
        assert retrieve_retry_pos < retry_results_pos, (
            f"retry poll (pos {retrieve_retry_pos}) must precede retry results "
            f"(pos {retry_results_pos}); call_log: {call_log}"
        )
        assert results["sess_1"] == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_retry_submit_uses_save_pending_false(self):
        """Retry's submit_batch must be called with save_pending=False.

        Issue #2: if save_pending=True (the default), the retry overwrites the main
        batch's pending_batch.json with a file containing only the retry sessions.
        A crash mid-retry then leaves the wrong batch ID on disk for resume.
        This test verifies the flag is passed, not just that the final state is clean
        (the end-of-function cleanup would make the final state clean regardless).
        """
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MagicMock()
            client.messages.batches.retrieve.return_value = self._ended_status()
            client.messages.batches.results.side_effect = [
                iter([self._make_success("sess_1", "P,S,U,EXTRA")]),  # parse failure
                iter([self._make_success("sess_1", "P,S,U")]),  # retry ok
            ]
            client.messages.batches.create.return_value = SimpleNamespace(
                id="retry_batch_id"
            )

            with patch(
                "src.pipeline.batch_label._get_client", return_value=client
            ), patch(
                "src.pipeline.batch_label.submit_batch", return_value="retry_batch_id"
            ) as mock_submit:
                from src.pipeline.batch_label import poll_and_retrieve

                poll_and_retrieve(
                    "main_batch",
                    "nlile",
                    tmpdir,
                    {"sess_1": ("transcript", 3)},
                )

            mock_submit.assert_called_once()
            assert mock_submit.call_args.kwargs.get("save_pending") is False, (
                "retry submit_batch must pass save_pending=False to avoid "
                "clobbering the main batch's pending file"
            )


class TestDryRunEstimate:
    def test_dry_run_does_not_exit_process(self):
        """--dry-run-estimate must return {} without calling sys.exit.

        Bug: run_batch_label called sys.exit(0) after printing the estimate,
        so multi-source dry-runs exited after the first source.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = [
                {"session_id": f"sess_{i}", "steps": [{}] * 35} for i in range(3)
            ]
            from src.pipeline.batch_label import run_batch_label

            # Should return {} without raising SystemExit
            result = run_batch_label(
                source_dir="datasets/nlile/",
                raw_sessions=sessions,
                labels_dir=tmpdir,
                dry_run_estimate=True,
            )
            assert result == {}
