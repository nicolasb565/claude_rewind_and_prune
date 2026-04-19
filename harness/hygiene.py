"""Context hygiene operations applied when checkpoint_progress is emitted.

The agent's `messages` list is the source of truth for each next request.
When a checkpoint fires, we mutate earlier tool_result messages in place —
replacing their content with a short elision marker that references the
checkpoint. This preserves the tool_use ↔ tool_result pairing required
by the chat template while compressing the earlier tool outputs.
"""
from __future__ import annotations


ELIDE_MARKER = "[elided — see checkpoint: {finding}]"


def apply_checkpoint_elision(
    messages: list[dict],
    checkpoint_args: dict,
    keep_last_n_tool_results: int = 2,
) -> int:
    """Replace tool_result contents older than `keep_last_n_tool_results`
    with an elision marker referencing the checkpoint.

    Returns the number of tool_results that were elided.
    """
    finding = checkpoint_args.get("finding", "")[:140]
    marker = ELIDE_MARKER.format(finding=finding or "prior exploration")

    # Walk tool_result messages from newest to oldest; keep the first N
    # (which are the newest), elide the rest if not already elided.
    tool_idxs = [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool"
    ]

    # Protect the most recent N tool_results (by index order)
    to_elide = tool_idxs[:-keep_last_n_tool_results] if len(tool_idxs) > keep_last_n_tool_results else []

    n_elided = 0
    for i in to_elide:
        m = messages[i]
        if m.get("content") == marker or m.get("content", "").startswith("[elided"):
            continue  # already elided by a prior checkpoint
        m["content"] = marker
        n_elided += 1
    return n_elided
