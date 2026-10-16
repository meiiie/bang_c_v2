#!/usr/bin/env python3
"""Run mock mode; report getrusage peaks and sampled aggregate Linux process RSS."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_tree_rss():
    """Sample all live descendants, including concurrently running program workers."""
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts = path.read_text().rsplit(')', 1)[1].split()
            processes[int(path.parent.name)] = (int(parts[1]), int(parts[21]) * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {os.getpid()}
    while True:
        added = {pid for pid, (parent, _) in processes.items() if parent in descendants} - descendants
        if not added:
            break
        descendants.update(added)
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT.parent/'data'))
    args = parser.parse_args()
    peak, stopped = [0], threading.Event()
    def sample():
        while not stopped.is_set():
            peak[0] = max(peak[0], process_tree_rss())
            stopped.wait(.02)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        from main import main as solve
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'mock.csv'
            solve(['--input', args.input, '--output', str(output), '--mode', 'mock', '--profile', 'contest'])
            # Validate accuracy externally below too; do not let a fallback-only run pass unnoticed.
            import csv
            with output.open(newline='') as stream:
                predicted = list(csv.DictReader(stream))
            with (Path(args.input)/'sample_submission.csv').open(encoding='utf-8-sig', newline='') as stream:
                gold = list(csv.DictReader(stream))
            assert predicted == gold, 'Mock answers differ from gold'
    finally:
        stopped.set()
        monitor.join()
    scale = 1 if sys.platform == 'darwin' else 1024
    main_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    report = {'main_peak_rss_bytes': main_peak, 'largest_child_peak_rss_bytes': child_peak,
              'main_plus_largest_child_bytes': main_peak + child_peak,
              'sampled_main_plus_all_children_bytes': peak[0]}
    print(json.dumps(report, indent=2))
    assert max(main_peak + child_peak, peak[0]) < 1024**3, 'RSS exceeds 1 GiB'


if __name__ == '__main__':
    main()
