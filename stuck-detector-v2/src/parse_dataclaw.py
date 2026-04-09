"""Parse DataClaw conversations.jsonl into step dicts for abstract_trajectory.

DataClaw has two formats:
  - woctordho: tool_uses with {tool, input(dict), output, status}
  - peteromallet: tool_uses with {tool, input(str)} — NO outputs (skip these)

Each step dict matches the schema expected by abstract_trajectory:
  {tool, cmd, file, output, thinking}
"""

# Tool name → abstract category (same as abstract_trajectory.TOOL_TO_IDX)
TOOL_MAP = {
    'Bash': 'bash', 'bash': 'bash',
    'Read': 'view', 'read': 'view',
    'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
    'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
    'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
}


def _extract_output(tu):
    """Extract output text from a tool_use dict."""
    out = tu.get('output', '')
    if isinstance(out, dict):
        return out.get('text', str(out))
    return str(out) if out else ''


def _extract_input_fields(tool_name, inp):
    """Extract cmd and file_path from tool input."""
    if isinstance(inp, dict):
        cmd = inp.get('command', inp.get('pattern', ''))
        file_path = inp.get('file_path', inp.get('path', None))
        # For Read/Edit/Write, use file_path as cmd if no command
        if not cmd and file_path:
            cmd = file_path
        elif not cmd and not file_path:
            # Agent/Task: extract description or prompt
            cmd = inp.get('description', inp.get('prompt', ''))[:200]
        return str(cmd), file_path
    # String input (peteromallet format)
    return str(inp)[:200] if inp else '', None


def parse_dataclaw_session(messages):
    """Parse a DataClaw session's messages into step dicts.

    Returns list of {tool, cmd, file, output, thinking}.
    Only includes sessions that have tool outputs.
    """
    steps = []
    pending_thinking = None

    for msg in messages:
        role = msg.get('role', '')

        # Collect thinking
        if 'thinking' in msg and msg['thinking']:
            pending_thinking = msg['thinking']
            continue

        # Process tool uses
        if 'tool_uses' not in msg:
            continue

        for tu in msg['tool_uses']:
            tool_name = tu.get('tool', 'other')
            tool = TOOL_MAP.get(tool_name, 'other')

            # Skip if no output field at all (peteromallet format)
            if 'output' not in tu and 'status' not in tu:
                continue

            inp = tu.get('input', '')
            cmd, file_path = _extract_input_fields(tool_name, inp)
            output = _extract_output(tu)

            # Check error status
            status = tu.get('status', '')
            if status == 'error' and output and not output.startswith('Error'):
                output = f"Error: {output}"

            steps.append({
                'tool': tool,
                'cmd': cmd,
                'file': file_path,
                'output': output,
                'thinking': pending_thinking or '',
            })
            pending_thinking = None

    return steps


def has_outputs(messages):
    """Check if a session has tool outputs (woctordho format)."""
    for msg in messages:
        if 'tool_uses' in msg:
            for tu in msg['tool_uses']:
                if 'output' in tu:
                    return True
    return False
