"""Tests for src/pipeline/label_session.py."""

import json
import os
import tempfile

import pytest

from src.pipeline.label_session import (
    format_transcript,
    parse_csv_labels,
    validate_label_file,
    write_label_file,
)


def _make_step(tool_name="Bash", cmd="ls", output="file.txt", thinking="") -> dict:
    return {
        "tool": "bash",
        "tool_name": tool_name,
        "cmd": cmd,
        "file": None,
        "output": output,
        "thinking": thinking,
    }


class TestFormatTranscript:
    def test_output_capped_at_500_chars(self):
        long_output = "x" * 600
        steps = [_make_step(output=long_output)]
        transcript, n = format_transcript(steps)
        # Should be truncated to 500 + "[...]" = 503 chars total for output portion
        assert "[...]" in transcript
        # Verify the actual output in transcript doesn't exceed 503
        output_part = long_output[:500] + "[...]"
        assert output_part in transcript

    def test_output_under_500_no_truncation_marker(self):
        short_output = "x" * 400
        steps = [_make_step(output=short_output)]
        transcript, n = format_transcript(steps)
        assert "[...]" not in transcript
        assert short_output in transcript

    def test_output_exactly_500_no_truncation(self):
        output_500 = "y" * 500
        steps = [_make_step(output=output_500)]
        transcript, _ = format_transcript(steps)
        assert "[...]" not in transcript

    def test_correct_step_count(self):
        steps = [_make_step() for _ in range(5)]
        _, n = format_transcript(steps)
        assert n == 5

    def test_compact_blocks_not_counted_as_steps(self):
        items = [
            {"type": "compact", "text": "summary block"},
            _make_step(),
            _make_step(),
        ]
        _, n = format_transcript(items)
        assert n == 2

    def test_compact_block_rendered(self):
        items = [
            {"type": "compact", "text": "some prior context"},
            _make_step(),
        ]
        transcript, _ = format_transcript(items)
        assert "[compact:" in transcript
        assert "some prior context" in transcript

    def test_compact_text_truncated_to_300(self):
        # Use a text where the tail after 300 chars is uniquely identifiable
        long_text = "a" * 295 + "UNIQUE_TAIL_TEXT_XYZ"
        items = [{"type": "compact", "text": long_text}]
        transcript, _ = format_transcript(items)
        assert "a" * 295 in transcript
        assert "UNIQUE_TAIL_TEXT_XYZ" not in transcript

    def test_step_numbering(self):
        steps = [_make_step() for _ in range(3)]
        transcript, _ = format_transcript(steps)
        assert "[0]" in transcript
        assert "[1]" in transcript
        assert "[2]" in transcript

    def test_total_steps_appended(self):
        steps = [_make_step() for _ in range(4)]
        transcript, n = format_transcript(steps)
        assert f"Total steps: {n}" in transcript
        assert n == 4

    def test_bash_command_rendered(self):
        step = _make_step(tool_name="Bash", cmd="make -j8")
        transcript, _ = format_transcript([step])
        assert "command: make -j8" in transcript

    def test_read_file_path_rendered(self):
        step = _make_step(tool_name="Read", cmd="src/main.c")
        transcript, _ = format_transcript([step])
        assert "file_path: src/main.c" in transcript


class TestValidateLabelFile:
    def test_valid_file_returns_true(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"labels": ["PRODUCTIVE", "STUCK", "UNSURE"], "n_steps": 3}, f)
            path = f.name
        try:
            assert validate_label_file(path, 3) is True
        finally:
            os.unlink(path)

    def test_wrong_count_returns_false(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"labels": ["PRODUCTIVE", "STUCK"], "n_steps": 2}, f)
            path = f.name
        try:
            assert validate_label_file(path, 5) is False
        finally:
            os.unlink(path)

    def test_invalid_json_returns_false(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("not json {{{")
            path = f.name
        try:
            assert validate_label_file(path, 3) is False
        finally:
            os.unlink(path)

    def test_missing_file_returns_false(self):
        assert validate_label_file("/nonexistent/path.json", 3) is False


class TestWriteLabelFile:
    def test_writes_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_labels.json")
            labels = ["PRODUCTIVE", "STUCK", "UNSURE"]
            write_label_file(path, "sess_001", "nlile", labels, 3)
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "sess_001"
            assert data["source"] == "nlile"
            assert data["n_steps"] == 3
            assert data["labels"] == labels
            assert "labeled_at" in data
            assert "labeler" in data


class TestParseCsvLabels:
    def test_upper_case(self):
        result = parse_csv_labels("P,S,U", 3)
        assert result == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_lower_case(self):
        result = parse_csv_labels("p,s,u", 3)
        assert result == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_mixed_case(self):
        result = parse_csv_labels("P,s,U,p", 4)
        assert result == ["PRODUCTIVE", "STUCK", "UNSURE", "PRODUCTIVE"]

    def test_whitespace_stripped(self):
        result = parse_csv_labels(" P , S , U ", 3)
        assert result == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_trailing_comma_handled(self):
        result = parse_csv_labels("P,S,U,", 3)
        assert result == ["PRODUCTIVE", "STUCK", "UNSURE"]

    def test_unknown_char_raises(self):
        with pytest.raises(ValueError, match="Unknown label"):
            parse_csv_labels("P,X,U", 3)

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            parse_csv_labels("P,S", 5)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            parse_csv_labels("", 3)

    def test_any_count_mismatch_raises(self):
        """Any label count != n_steps must raise — callers handle retry logic."""
        # (640, 646): real production incident — 6 labels missing (genuine miscount)
        # (57, 58):   real production incident — 1 label missing (off-by-one)
        # (512, 646): real production incident — max_tokens truncation (now fixed)
        for got, expected in [(640, 646), (57, 58), (512, 646)]:
            csv = ",".join(["P"] * got)
            with pytest.raises(ValueError, match="mismatch"):
                parse_csv_labels(csv, expected)

    def test_empty_middle_part_raises(self):
        """Empty slot in the middle must not be silently dropped.

        With n_steps=3 and "P,,S,U" (4 parts, 1 empty), the filter currently
        compresses to 3 labels and silently accepts — masking a genuine miscount.
        """
        with pytest.raises(ValueError, match="empty"):
            parse_csv_labels("P,,S,U", 3)
