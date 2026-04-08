"""Parser for ADP standardized format (neulab/agent-data-collection).

Covers: nebius_SWE-agent-trajectories, swe-smith, swe-gym_openhands_sampled_trajectories,
        agenttuning_webshop.

ADP format: each trajectory has 'content' (list of steps) where steps alternate between
action (class_=api_action|code_action|message_action) and observation (class_=text_observation).

Outputs a list of (tool, cmd, file, output, thinking) tuples — one per action+observation pair.
"""

import json
import re


def categorize_tool(class_, function=None, content=None):
    """Map ADP action to abstract tool category."""
    if class_ == 'code_action':
        return 'bash'
    elif class_ == 'api_action':
        if function in ('find_file', 'search_dir', 'search_file', 'grep'):
            return 'search'
        elif function in ('open_file', 'goto', 'scroll_down', 'scroll_up', 'cat', 'view'):
            return 'view'
        elif function in ('edit_file', 'str_replace_editor', 'insert', 'replace'):
            return 'edit'
        elif function in ('create_file',):
            return 'create'
        elif function == 'submit':
            return 'submit'
        else:
            return 'other'
    elif class_ == 'message_action':
        return 'other'
    return 'other'


def extract_file_from_kwargs(kwargs, function=None):
    """Extract target file path from ADP kwargs."""
    if not kwargs:
        return None
    for key in ('path', 'file_name', 'file', 'filename'):
        if key in kwargs:
            val = kwargs[key]
            if isinstance(val, str) and val:
                return val
    return None


def extract_command(step):
    """Build a command string from an ADP step for hashing."""
    cls = step.get('class_', '')
    if cls == 'code_action':
        return step.get('content', '')
    elif cls == 'api_action':
        fn = step.get('function', '')
        kwargs = step.get('kwargs', {})
        return f'{fn}({json.dumps(kwargs, sort_keys=True)})'
    elif cls == 'message_action':
        return step.get('content', '')[:200]
    return ''


def parse_adp_trajectory(traj):
    """Parse one ADP trajectory into a list of abstract step dicts.

    Returns list of dicts with keys: tool, cmd, file, output, thinking
    """
    content = traj.get('content', [])
    steps = []

    i = 0
    while i < len(content):
        step = content[i]
        cls = step.get('class_', '')

        if cls in ('api_action', 'code_action', 'message_action'):
            tool = categorize_tool(cls, step.get('function'), step.get('content'))
            cmd = extract_command(step)
            file = extract_file_from_kwargs(step.get('kwargs'), step.get('function'))
            thinking = step.get('description', '') or ''

            # Next step should be the observation
            output = ''
            if i + 1 < len(content) and content[i + 1].get('class_') == 'text_observation':
                output = content[i + 1].get('content', '') or ''
                i += 2
            else:
                i += 1

            steps.append({
                'tool': tool,
                'cmd': cmd,
                'file': file,
                'output': output,
                'thinking': thinking,
            })
        else:
            # Skip observations without a preceding action
            i += 1

    return steps


def parse_adp_metadata(traj, subset_name):
    """Extract metadata from an ADP trajectory."""
    details = traj.get('details', {})
    return {
        'trajectory_id': traj.get('id', ''),
        'source_dataset': f'adp/{subset_name}',
        'exit_status': details.get('exit_status'),
        'resolved': details.get('resolved'),
        'test_result': details.get('test_result'),
    }


def iter_adp_file(path, subset_name):
    """Iterate over an ADP JSONL file, yielding (parsed_steps, metadata) tuples."""
    with open(path) as f:
        for line in f:
            traj = json.loads(line)
            steps = parse_adp_trajectory(traj)
            meta = parse_adp_metadata(traj, subset_name)
            yield steps, meta
