"""Tool execution for the agent harness.

Each tool takes the arguments dict parsed from a Qwen tool call and
returns a string (the tool_result content). All tools operate inside a
working directory passed at construction time — this is the sandbox.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path


MAX_OUTPUT_CHARS = 8000  # truncate tool outputs past this
READ_MAX_LINES = 500


class ToolRunner:
    def __init__(self, work_dir: Path, timeout_s: int = 30):
        self.work_dir = Path(work_dir).resolve()
        self.timeout_s = timeout_s

    def run(self, name: str, args: dict) -> str:
        handler = {
            "Bash": self._bash,
            "Read": self._read,
            "Write": self._write,
            "Edit": self._edit,
            "Grep": self._grep,
            "Glob": self._glob,
        }.get(name)
        if handler is None:
            return f"[harness] unknown tool: {name}"
        try:
            return _truncate(handler(args))
        except Exception as e:
            return f"[harness] tool {name} raised {type(e).__name__}: {e}"

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.work_dir / p
        p = p.resolve()
        # Sandbox: refuse paths outside work_dir
        if self.work_dir not in p.parents and p != self.work_dir:
            raise ValueError(f"path outside sandbox: {p}")
        return p

    def _bash(self, args: dict) -> str:
        cmd = args.get("command", "").strip()
        if not cmd:
            return "[harness] Bash: empty command"
        result = subprocess.run(
            cmd, shell=True, cwd=str(self.work_dir),
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts)

    def _read(self, args: dict) -> str:
        path = args.get("file_path", "").strip()
        if not path:
            return "[harness] Read: missing file_path"
        p = self._resolve(path)
        if not p.exists():
            return f"[harness] Read: not found: {path}"
        lines = p.read_text().splitlines()
        if len(lines) > READ_MAX_LINES:
            lines = lines[:READ_MAX_LINES] + [f"... [truncated at {READ_MAX_LINES} lines]"]
        return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

    def _write(self, args: dict) -> str:
        path = args.get("file_path", "").strip()
        content = args.get("content", "")
        if not path:
            return "[harness] Write: missing file_path"
        if content == "[elided]" or content.strip() == "":
            return "[harness] Write: content was elided or empty, refusing"
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {p.relative_to(self.work_dir)} ({len(content)} chars)"

    def _edit(self, args: dict) -> str:
        """Edit accepts either old_string/new_string (like real Claude Code)
        or a content replacement. Model variants differ."""
        path = args.get("file_path", "").strip()
        if not path:
            return "[harness] Edit: missing file_path"
        p = self._resolve(path)
        if not p.exists():
            return f"[harness] Edit: file not found: {path}"

        old_string = args.get("old_string")
        new_string = args.get("new_string")
        content = args.get("content")

        if old_string is not None and new_string is not None:
            text = p.read_text()
            if old_string not in text:
                return f"[harness] Edit: old_string not found in {path}"
            if text.count(old_string) > 1:
                return f"[harness] Edit: old_string matches {text.count(old_string)} places, must be unique"
            new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text)
            return f"Edited {p.relative_to(self.work_dir)} (replaced 1 occurrence)"

        if content is not None and content != "[elided]":
            p.write_text(content)
            return f"Wrote full file {p.relative_to(self.work_dir)} ({len(content)} chars)"

        return "[harness] Edit: need old_string+new_string or content; got neither"

    def _grep(self, args: dict) -> str:
        pattern = args.get("pattern", "").strip()
        path = args.get("path", ".")
        if not pattern:
            return "[harness] Grep: missing pattern"
        p = self._resolve(path) if path else self.work_dir
        result = subprocess.run(
            ["grep", "-rn", "--", pattern, str(p)],
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        if result.returncode == 1:  # grep returns 1 on no matches
            return "[no matches]"
        return result.stdout or f"[exit {result.returncode}]"

    def _glob(self, args: dict) -> str:
        pattern = args.get("pattern", "").strip()
        if not pattern:
            return "[harness] Glob: missing pattern"
        matches = sorted(str(p.relative_to(self.work_dir))
                         for p in self.work_dir.rglob(pattern))
        return "\n".join(matches) if matches else "[no matches]"


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return s[:half] + f"\n... [truncated, total {len(s)} chars]\n" + s[-half:]
