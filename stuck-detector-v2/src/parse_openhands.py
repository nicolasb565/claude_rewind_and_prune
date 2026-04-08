"""Parser for OpenHands message format.

Covers: MEnvData, Nebius OpenHands, Nemotron.

Two sub-formats:
1. Structured: assistant messages have tool_calls array, tool messages have results (Nebius OH, Nemotron)
2. XML-embedded: tool calls are in assistant content as <function=name><parameter=...>...</function> (MEnvData)

Outputs a list of (tool, cmd, file, output, thinking) tuples — one per tool call+result pair.
"""

import json
import re


TOOL_CATEGORY_MAP = {
    'execute_bash': 'bash',
    'str_replace_editor': 'edit',
    'create': 'create',
    'view': 'view',
    'find_file': 'search',
    'search_dir': 'search',
    'search_file': 'search',
    'grep': 'search',
    'think': 'other',
    'submit': 'submit',
}


def categorize_tool(name):
    """Map OpenHands tool name to abstract category."""
    if not name:
        return 'other'
    if name in TOOL_CATEGORY_MAP:
        return TOOL_CATEGORY_MAP[name]
    name_lower = name.lower()
    if 'bash' in name_lower or 'execute' in name_lower:
        return 'bash'
    if 'edit' in name_lower or 'replace' in name_lower or 'write' in name_lower:
        return 'edit'
    if 'view' in name_lower or 'read' in name_lower or 'cat' in name_lower:
        return 'view'
    if 'search' in name_lower or 'find' in name_lower or 'grep' in name_lower:
        return 'search'
    return 'other'


def extract_file_from_args(args_str, tool_name=None):
    """Extract target file from tool call arguments."""
    if not args_str:
        return None
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(args, dict):
        for key in ('path', 'file_name', 'file', 'filename'):
            if key in args:
                val = args[key]
                if isinstance(val, str) and val:
                    return val
    return None


def extract_command(tool_name, args_str):
    """Build a command string from tool call for hashing."""
    if not args_str:
        return tool_name or ''
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except (json.JSONDecodeError, TypeError):
        return f'{tool_name}({args_str[:200]})'

    if isinstance(args, dict):
        if 'command' in args:
            return args['command']
        return f'{tool_name}({json.dumps(args, sort_keys=True)})'
    return f'{tool_name}({str(args)[:200]})'


def _parse_xml_function_calls(content):
    """Parse XML-style function calls from MEnvData assistant content.

    Format: <function=tool_name>
    <parameter=param_name>value</parameter>
    </function>

    Returns (thinking, list of {name, args_dict}) tuples.
    """
    # Split content into thinking (before first <functions>) and function calls
    parts = re.split(r'<functions>\s*', content, maxsplit=1)
    thinking = parts[0].strip() if parts else ''
    if len(parts) < 2:
        return thinking, []

    fn_text = parts[1]
    calls = []
    for fn_match in re.finditer(r'<function=(\w+)>(.*?)</function>', fn_text, re.DOTALL):
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)
        args = {}
        for param_match in re.finditer(
                r'<parameter=(\w+)>(.*?)</parameter>', fn_body, re.DOTALL):
            args[param_match.group(1)] = param_match.group(2).strip()
        calls.append({'name': fn_name, 'args': args})

    return thinking, calls


def _detect_xml_format(messages):
    """Check if messages use XML-embedded function calls (MEnvData style)."""
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '') or ''
            if '<function=' in content:
                return True
            if msg.get('tool_calls'):
                return False
    return False


def parse_openhands_trajectory(messages):
    """Parse OpenHands messages into a list of abstract step dicts.

    Auto-detects structured vs XML-embedded format.
    Returns list of dicts with keys: tool, cmd, file, output, thinking
    """
    if _detect_xml_format(messages):
        return _parse_xml_trajectory(messages)
    return _parse_structured_trajectory(messages)


def _parse_xml_trajectory(messages):
    """Parse XML-embedded format (MEnvData)."""
    steps = []

    # Collect tool results in order — MEnvData doesn't use tool_call_ids,
    # tool messages follow the assistant message sequentially
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get('role') != 'assistant':
            i += 1
            continue

        content = msg.get('content', '') or ''
        reasoning = msg.get('reasoning_content', '') or ''

        thinking, fn_calls = _parse_xml_function_calls(content)
        if reasoning:
            thinking = reasoning + '\n' + thinking

        if not fn_calls:
            i += 1
            continue

        # Collect subsequent tool messages as results
        tool_outputs = []
        j = i + 1
        while j < len(messages) and messages[j].get('role') == 'tool':
            tool_outputs.append(messages[j].get('content', ''))
            j += 1

        for k, call in enumerate(fn_calls):
            fn_name = call['name']
            args = call['args']

            tool = categorize_tool(fn_name)
            cmd = extract_command(fn_name, json.dumps(args, sort_keys=True))
            file = extract_file_from_args(args, fn_name)
            output = tool_outputs[k] if k < len(tool_outputs) else ''

            if fn_name == 'str_replace_editor':
                subcmd = args.get('command', '')
                if subcmd == 'view':
                    tool = 'view'
                elif subcmd in ('str_replace', 'insert', 'create'):
                    tool = 'edit'

            steps.append({
                'tool': tool,
                'cmd': cmd,
                'file': file,
                'output': output,
                'thinking': thinking if k == 0 else '',
            })

        i = j

    return steps


def _parse_structured_trajectory(messages):
    """Parse structured format (Nebius OH, Nemotron)."""
    steps = []

    # Build a map from tool_call_id to tool result content
    tool_results = {}
    for msg in messages:
        if msg.get('role') == 'tool':
            tcid = msg.get('tool_call_id') or msg.get('id')
            if tcid:
                tool_results[tcid] = msg.get('content', '')

    for msg in messages:
        if msg.get('role') != 'assistant':
            continue

        thinking = msg.get('content', '') or ''
        # Some models put reasoning in a separate field
        reasoning = msg.get('reasoning_content', '') or ''
        if reasoning:
            thinking = reasoning + '\n' + thinking

        tool_calls = msg.get('tool_calls')
        if not tool_calls:
            continue

        for tc in tool_calls:
            fn = tc.get('function', {})
            tool_name = fn.get('name', '')
            args_str = fn.get('arguments', '')
            tc_id = tc.get('id', '')

            # Skip 'think' tool calls — they're just reasoning
            if tool_name == 'think':
                # Append think content to thinking for this step
                try:
                    think_args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    if isinstance(think_args, dict):
                        thinking += '\n' + think_args.get('thought', '')
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            tool = categorize_tool(tool_name)
            cmd = extract_command(tool_name, args_str)
            file = extract_file_from_args(args_str, tool_name)
            output = tool_results.get(tc_id, '')

            # For str_replace_editor, the 'command' arg determines if it's edit or view
            if tool_name == 'str_replace_editor':
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    if isinstance(args, dict):
                        subcmd = args.get('command', '')
                        if subcmd == 'view':
                            tool = 'view'
                        elif subcmd in ('str_replace', 'insert', 'create'):
                            tool = 'edit'
                except (json.JSONDecodeError, TypeError):
                    pass

            steps.append({
                'tool': tool,
                'cmd': cmd,
                'file': file,
                'output': output,
                'thinking': thinking,
            })
            # Only use thinking for the first tool call in a message
            thinking = ''

    return steps


def parse_openhands_metadata(row, source_name):
    """Extract metadata from an OpenHands dataset row."""
    meta = {
        'source_dataset': source_name,
    }

    # Different datasets have different fields
    if 'trajectory_id' in row:
        meta['trajectory_id'] = row['trajectory_id']
    elif 'uuid' in row:
        meta['trajectory_id'] = row['uuid']
    else:
        meta['trajectory_id'] = ''

    if 'instance_id' in row:
        meta['instance_id'] = row['instance_id']
    if 'repo' in row:
        meta['repo'] = row['repo']
    if 'resolved' in row:
        meta['resolved'] = bool(row['resolved'])
    if 'exit_status' in row:
        meta['exit_status'] = row['exit_status']
    if 'docker_image' in row:
        meta['docker_image'] = row['docker_image']
    if 'feedback' in row:
        meta['feedback'] = row['feedback']

    return meta


def iter_openhands_dataset(ds, source_name, messages_key='messages'):
    """Iterate over a HuggingFace dataset, yielding (parsed_steps, metadata) tuples."""
    for i in range(len(ds)):
        row = ds[i]
        messages = row.get(messages_key) or row.get('trajectory', [])
        steps = parse_openhands_trajectory(messages)
        meta = parse_openhands_metadata(row, source_name)
        yield steps, meta
