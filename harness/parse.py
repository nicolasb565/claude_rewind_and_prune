"""Parse Qwen 3.5's XML tool_call format into a structured dict.

Format emitted by the model:
    <tool_call>
    <function=Bash>
    <parameter=command>
    ls -la
    </parameter>
    </function>
    </tool_call>
"""
from __future__ import annotations

import re


_FUNCTION_RE = re.compile(
    r"<tool_call>\s*<function=([\w_]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> dict | None:
    """Return {'name': str, 'arguments': dict} or None if no tool call found.

    Returns the FIRST tool call if multiple appear. The model should emit
    one per turn but we don't enforce that here.
    """
    m = _FUNCTION_RE.search(text)
    if m is None:
        return None
    name = m.group(1)
    body = m.group(2)
    args = dict(_PARAMETER_RE.findall(body))
    return {"name": name, "arguments": args}


def is_checkpoint(tool_call: dict) -> bool:
    return tool_call.get("name") == "mcp__bookmarks__checkpoint_progress"
