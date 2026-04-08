"""Phase 2.5: Curate a balanced ~15-20K training pool.

Processes one dataset at a time to control memory usage.
Uses reservoir sampling for large datasets to avoid loading all into memory.
"""

import json
import random
import sys
import os
import gc
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from parse_adp import parse_adp_trajectory, parse_adp_metadata
from parse_openhands import parse_openhands_trajectory, parse_openhands_metadata
from abstract_trajectory import abstract_trajectory

random.seed(42)

POOL_FILE = 'data/curated_pool.jsonl'


def step_count_bin(n):
    if n < 10:
        return 'short'
    elif n < 30:
        return 'medium'
    else:
        return 'long'


def compute_repetition_score(parsed_steps):
    """Compute repetition score without full abstraction (lighter)."""
    import zlib
    if len(parsed_steps) < 3:
        return 0.0
    cmd_hashes = []
    for s in parsed_steps:
        cmd = s.get('cmd', '')
        cmd_hashes.append(zlib.crc32(cmd.encode()) if cmd else None)

    repeats = 0
    for i, h in enumerate(cmd_hashes):
        if h is None:
            continue
        for j in range(max(0, i - 3), i):
            if cmd_hashes[j] == h:
                repeats += 1
                break
    return repeats / len(cmd_hashes)


def reservoir_sample(iterator, n):
    """Reservoir sampling: select n items from an iterator of unknown length."""
    reservoir = []
    for i, item in enumerate(iterator):
        if i < n:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = item
    return reservoir


def stratified_sample(items, n, key='step_count_bin'):
    """Sample n items, stratified by a binning key."""
    if len(items) <= n:
        return list(items)
    bins = defaultdict(list)
    for item in items:
        bins[item.get(key, 'medium')].append(item)
    per_bin = max(1, n // len(bins))
    result = []
    for bin_items in bins.values():
        result.extend(random.sample(bin_items, min(per_bin, len(bin_items))))
    remaining = [item for item in items if item not in result]
    if len(result) < n and remaining:
        result.extend(random.sample(remaining, min(n - len(result), len(remaining))))
    return result[:n]


def make_record(traj_id, source, strength, parsed_steps, exit_status=None,
                resolved=None, repo=None, rep_score=None):
    """Create a pool record. Stores only the parsed steps, not raw data."""
    rec = {
        'trajectory_id': traj_id,
        'source_dataset': source,
        'model_strength': strength,
        'n_steps': len(parsed_steps),
        'step_count_bin': step_count_bin(len(parsed_steps)),
        'parsed_steps': parsed_steps,
    }
    if exit_status is not None:
        rec['exit_status'] = exit_status
    if resolved is not None:
        rec['resolved'] = resolved
    if repo is not None:
        rec['repo'] = repo
    if rep_score is not None:
        rec['repetition_score'] = rep_score
    return rec


def save_records(records, label):
    """Append records to pool file and print summary."""
    with open(POOL_FILE, 'a') as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + '\n')
    print(f"  Saved {len(records)} ({label})")


def process_menvdata():
    """MEnvData: all trajectories (irreplaceable, multi-language, Claude)."""
    print("\n=== MEnvData (strong, all included) ===")
    from datasets import load_from_disk
    ds = load_from_disk('data/separate/menvdata')
    records = []
    for i in range(len(ds)):
        row = ds[i]
        parsed = parse_openhands_trajectory(row['messages'])
        if len(parsed) < 3:
            continue
        records.append(make_record(
            f'menvdata_{i}', 'menvdata', 'strong', parsed,
            repo=row.get('docker_image', '')
        ))
    save_records(records, 'menvdata')
    count = len(records)
    del ds, records
    gc.collect()
    return count


def process_nebius_openhands():
    """Nebius OpenHands: sample 3000, balanced resolved/failed."""
    print("\n=== Nebius OpenHands (strong, sample 3000) ===")
    from datasets import load_from_disk
    ds = load_from_disk('data/separate/nebius_openhands')

    resolved = []
    failed = []
    for i in range(len(ds)):
        row = ds[i]
        parsed = parse_openhands_trajectory(row['trajectory'])
        if len(parsed) < 3:
            continue
        rec = make_record(
            row['trajectory_id'], 'nebius_openhands', 'strong', parsed,
            exit_status=row.get('exit_status'),
            resolved=bool(row['resolved']),
            repo=row.get('repo')
        )
        if row['resolved']:
            resolved.append(rec)
        else:
            failed.append(rec)

        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(ds)}...")

    sample = stratified_sample(resolved, 1500) + stratified_sample(failed, 1500)
    save_records(sample, f'nebius_oh ({len(resolved)} resolved, {len(failed)} failed)')
    count = len(sample)
    del ds, resolved, failed, sample
    gc.collect()
    return count


def process_nemotron():
    """Nemotron: sample 1500."""
    print("\n=== Nemotron (strong, sample 1500) ===")
    from datasets import load_from_disk
    ds = load_from_disk('data/separate/nemotron')

    records = []
    for i in range(len(ds)):
        row = ds[i]
        parsed = parse_openhands_trajectory(row['messages'])
        if len(parsed) < 3:
            continue
        records.append(make_record(
            row['uuid'], 'nemotron', 'strong', parsed,
            repo=row.get('repo')
        ))
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(ds)}...")

    sample = stratified_sample(records, 1500)
    save_records(sample, f'nemotron (from {len(records)})')
    count = len(sample)
    del ds, records, sample
    gc.collect()
    return count


def process_adp_subset(subset_name, strength, target_n=None):
    """Process an ADP subset. If target_n given, sample down."""
    print(f"\n=== ADP {subset_name} ({strength}) ===")
    path = f'data/adp/{subset_name}/full_std.jsonl'
    records = []
    with open(path) as f:
        for line in f:
            traj = json.loads(line)
            parsed = parse_adp_trajectory(traj)
            if len(parsed) < 3:
                continue
            meta = parse_adp_metadata(traj, subset_name)
            records.append(make_record(
                meta['trajectory_id'], f'adp/{subset_name}', strength, parsed,
                exit_status=meta.get('exit_status'),
                resolved=meta.get('resolved')
            ))

    if target_n and len(records) > target_n:
        sample = stratified_sample(records, target_n)
    else:
        sample = records

    save_records(sample, f'{subset_name} (from {len(records)})')
    count = len(sample)
    del records, sample
    gc.collect()
    return count


def process_nebius_adp():
    """Nebius ADP (weak): stratified by outcome and repetition score."""
    print("\n=== ADP nebius (weak, stratified sampling) ===")
    path = 'data/adp/nebius_SWE-agent-trajectories/full_std.jsonl'

    resolved = []
    low_rep = []
    mid_rep = []
    high_rep = []

    with open(path) as f:
        for line in f:
            traj = json.loads(line)
            parsed = parse_adp_trajectory(traj)
            if len(parsed) < 3:
                continue
            meta = parse_adp_metadata(traj, 'nebius_SWE-agent-trajectories')
            rep = compute_repetition_score(parsed)
            rec = make_record(
                meta['trajectory_id'], 'adp/nebius_SWE-agent-trajectories',
                'weak', parsed,
                exit_status=meta.get('exit_status'),
                rep_score=rep
            )

            if meta.get('exit_status') == 'submitted':
                resolved.append(rec)
            elif rep <= 0.3:
                low_rep.append(rec)
            elif rep <= 0.7:
                mid_rep.append(rec)
            else:
                high_rep.append(rec)

    sample = (
        stratified_sample(resolved, 1000) +
        stratified_sample(low_rep, 800) +
        stratified_sample(mid_rep, 400) +
        stratified_sample(high_rep, 300)
    )

    save_records(sample, f'nebius ({len(resolved)} res, {len(low_rep)} low, '
                 f'{len(mid_rep)} mid, {len(high_rep)} high rep)')
    count = len(sample)
    del resolved, low_rep, mid_rep, high_rep, sample
    gc.collect()
    return count


def curate():
    # Clear pool file
    open(POOL_FILE, 'w').close()

    counts = {}

    # Process one at a time, free memory between each
    counts['menvdata'] = process_menvdata()
    counts['nebius_oh'] = process_nebius_openhands()
    counts['nemotron'] = process_nemotron()
    counts['swe_smith'] = process_adp_subset('swe-smith', 'medium', target_n=1500)
    counts['swe_gym'] = process_adp_subset('swe-gym_openhands_sampled_trajectories', 'medium')
    counts['webshop'] = process_adp_subset('agenttuning_webshop', 'medium')
    counts['nebius_adp'] = process_nebius_adp()

    # Summary
    total = sum(counts.values())
    print(f"\n{'='*50}")
    print(f"Total curated pool: {total}")
    for source, n in counts.items():
        print(f"  {source}: {n}")

    # Verify from file
    strength_counts = Counter()
    source_counts = Counter()
    step_bins = Counter()
    with open(POOL_FILE) as f:
        for line in f:
            rec = json.loads(line)
            strength_counts[rec['model_strength']] += 1
            source_counts[rec['source_dataset']] += 1
            step_bins[rec['step_count_bin']] += 1

    print(f"\nBy model strength:")
    for s in ['strong', 'medium', 'weak']:
        c = strength_counts[s]
        print(f"  {s}: {c} ({c/total:.0%})")

    print(f"\nBy step count:")
    for b in ['short', 'medium', 'long']:
        print(f"  {b}: {step_bins[b]}")

    print(f"\nBy source:")
    for src, c in source_counts.most_common():
        print(f"  {src}: {c}")

    print(f"\nPool saved to {POOL_FILE}")


if __name__ == '__main__':
    curate()
