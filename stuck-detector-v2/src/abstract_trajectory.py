"""Convert parsed steps into language-agnostic abstract feature sequences.

Each step becomes a dict of numeric features suitable for CNN input.
File paths and commands are CRC32-hashed. Output similarity uses Jaccard on normalized line sets.
"""

import math
import re
import zlib

WINDOW_SIZE = 10
STRIDE = 5
MAX_OUTPUT_LINES = 100

TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']
TOOL_TO_IDX = {t: i for i, t in enumerate(TOOL_NAMES)}


# --- Output normalization ---

def normalize_to_set(output):
    """Normalize output to a frozenset of lines for Jaccard comparison.
    Called once per output. The frozenset is reused for per-step and window-level features."""
    if not output:
        return frozenset()
    lines = output.strip().split('\n')[:MAX_OUTPUT_LINES]
    normalized = set()
    for line in lines:
        line = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', line)
        line = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', line)
        line = re.sub(r'pid[=: ]\d+', 'pid=PID', line, flags=re.I)
        line = re.sub(r'/tmp/[^\s]+', '/tmp/TMPFILE', line)
        line = re.sub(r'\d+\.\d{3,}s', 'N.NNNs', line)
        line = line.strip()
        if line:
            normalized.add(line)
    return frozenset(normalized)


def jaccard(current_set, previous_set):
    """Jaccard similarity between two frozensets."""
    if previous_set is None:
        return 0.5  # no comparison available — neutral midpoint
    if not current_set and not previous_set:
        return 1.0
    union = current_set | previous_set
    return len(current_set & previous_set) / len(union) if union else 1.0


# --- Thinking/reasoning features ---

FALSE_START_PATTERNS = re.compile(
    r'\b(actually|wait|hmm|let me reconsider|on second thought)\b', re.I)
STRATEGY_CHANGE_PATTERNS = re.compile(
    r'\b(different approach|try another|instead|alternatively|let me try a different)\b', re.I)
CIRCULAR_PATTERNS = re.compile(
    r'\b(try again|let me try|one more time|retry|attempt again)\b', re.I)

ERROR_PATTERNS = re.compile(
    r'(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied'
    r'|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError'
    r'|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError)',
    re.I)


def has_error_indicators(output):
    if not output:
        return False
    return bool(ERROR_PATTERNS.search(output[:2000]))


def word_overlap_similarity(text1, text2):
    """Simple word-overlap cosine similarity."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / math.sqrt(len(words1) * len(words2))


# --- History lookback ---

def steps_since(value, history):
    """How many steps back was this value last seen? len(history) if never."""
    if value is None:
        return len(history)
    for j in range(len(history) - 1, -1, -1):
        if history[j] == value:
            return len(history) - j
    return len(history)


def count_in(value, history):
    """How many times has this value appeared in history?"""
    if value is None:
        return 0
    return sum(1 for h in history if h == value)


# --- Main abstraction ---

def abstract_trajectory(parsed_steps):
    """Convert parsed steps into abstract feature sequence.

    Args:
        parsed_steps: list of dicts with keys: tool, cmd, file, output, thinking

    Returns:
        list of dicts, each with numeric features for CNN input
    """
    if not parsed_steps:
        return []

    total_steps = len(parsed_steps)
    abstract_seq = []

    tool_history = []
    file_hash_history = []
    cmd_hash_history = []
    output_history = {}  # cmd_hash -> frozenset
    prev_thinking = None

    for i, step in enumerate(parsed_steps):
        tool = step.get('tool', 'other')
        if tool not in TOOL_TO_IDX:
            tool = 'other'

        file_path = step.get('file')
        cmd = step.get('cmd', '')
        output = step.get('output', '')
        thinking = step.get('thinking', '')

        # Hash file and command
        file_hash = zlib.crc32(file_path.encode()) if file_path else None
        cmd_hash = zlib.crc32(cmd.encode()) if cmd else None

        # Output normalization and similarity
        output_set = normalize_to_set(output)
        output_sim = jaccard(output_set, output_history.get(cmd_hash))

        # Lookback features (normalized by total steps)
        norm = max(total_steps, 1)

        abstract_step = {
            # Categorical
            'tool': tool,
            'tool_idx': TOOL_TO_IDX[tool],

            # Cycle detection
            'steps_since_same_tool': steps_since(tool, tool_history) / norm,
            'steps_since_same_file': steps_since(file_hash, file_hash_history) / norm,
            'steps_since_same_cmd': steps_since(cmd_hash, cmd_hash_history) / norm,

            # Repetition counts
            'tool_count_in_window': count_in(tool, tool_history) / max(i + 1, 1),
            'file_count_in_window': count_in(file_hash, file_hash_history) / max(i + 1, 1),
            'cmd_count_in_window': count_in(cmd_hash, cmd_hash_history) / max(i + 1, 1),

            # Output
            'output_similarity': output_sim,
            'output_set': output_set,
            'output_length': math.log1p(output.count('\n') + 1 if output else 0),
            'is_error': has_error_indicators(output),
            'step_index_norm': i / max(total_steps - 1, 1),

            # Thinking
            'false_start': bool(FALSE_START_PATTERNS.search(thinking)) if thinking else False,
            'strategy_change': bool(STRATEGY_CHANGE_PATTERNS.search(thinking)) if thinking else False,
            'circular_lang': bool(CIRCULAR_PATTERNS.search(thinking)) if thinking else False,
            'thinking_length': math.log1p(len(thinking)),
            'self_similarity': word_overlap_similarity(thinking, prev_thinking),
        }
        abstract_seq.append(abstract_step)

        # Update histories
        tool_history.append(tool)
        file_hash_history.append(file_hash)
        cmd_hash_history.append(cmd_hash)
        if cmd_hash is not None:
            output_history[cmd_hash] = output_set
        prev_thinking = thinking

    return abstract_seq


def compute_window_features(window_steps):
    """Compute window-level aggregate features."""
    n = len(window_steps)
    if n == 0:
        return {}

    tools = [s['tool'] for s in window_steps]
    file_hashes = [s.get('file_hash') for s in window_steps
                   if 'file_hash' in s and s.get('file_hash') is not None]
    cmd_hashes = [s.get('cmd_hash') for s in window_steps
                  if 'cmd_hash' in s and s.get('cmd_hash') is not None]

    # For window features, use the lookback values already computed
    unique_tools_ratio = len(set(tools)) / n
    unique_files_ratio = len(set(file_hashes)) / max(len(file_hashes), 1) if file_hashes else 1.0
    unique_cmds_ratio = len(set(cmd_hashes)) / max(len(cmd_hashes), 1) if cmd_hashes else 1.0

    error_rate = sum(1 for s in window_steps if s['is_error']) / n
    output_sim_avg = sum(s['output_similarity'] for s in window_steps) / n

    # Cross-command output diversity
    all_lines = []
    for s in window_steps:
        output_set = s.get('output_set')
        if output_set:
            all_lines.extend(output_set)
    output_diversity = len(set(all_lines)) / max(len(all_lines), 1) if all_lines else 1.0

    return {
        'unique_tools_ratio': unique_tools_ratio,
        'unique_files_ratio': unique_files_ratio,
        'unique_cmds_ratio': unique_cmds_ratio,
        'error_rate': error_rate,
        'output_similarity_avg': output_sim_avg,
        'output_diversity': output_diversity,
    }


def precompute_review_counts(window_steps):
    """Pre-compute counts for the Haiku reviewer.

    These are included in batch data so Haiku doesn't have to count
    from raw step features (which it does unreliably).
    """
    tight_loop_steps = sum(
        1 for s in window_steps
        if s['steps_since_same_cmd'] < 0.15 and s['output_similarity'] > 0.8
    )
    diverse_steps = sum(
        1 for s in window_steps
        if s['steps_since_same_cmd'] > 0.5
    )
    error_steps = sum(1 for s in window_steps if s['is_error'])
    unique_tools = len(set(s['tool'] for s in window_steps))
    has_submit = any(s['tool'] == 'submit' for s in window_steps)

    return {
        'tight_loop_steps': tight_loop_steps,
        'diverse_steps': diverse_steps,
        'error_steps': error_steps,
        'unique_tools': unique_tools,
        'has_submit': has_submit,
    }
