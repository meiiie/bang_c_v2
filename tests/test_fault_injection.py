"""Offline CLI fault probes; subprocess bounds also catch broken deadline guards.

Only provider calls are replaced. Parsing, strategies, sandbox and CSV writes run
normally against ../data. Each probe has a hard outer timeout; no live run files
or environment-specific settings are reused.
"""
import csv
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAULTS = ('429', '500', 'timeout', '402', 'garbage', 'loop', 'memory', 'slow')
# Hand-checked emergency values for the public questions (including text/list
# questions, whose current solver fallback is a scalar zero).
FALLBACKS = ['0'] * 20
for index in (3, 10):
    FALLBACKS[index] = 'No'
FALLBACKS[6] = '0.00'
FALLBACKS[17] = '0, 0'        # q018: "two integers"
FALLBACKS[18] = '0.00, 0.00'  # q019: template `men, women`, two decimals


def _probe(kind, data_dir, directory, strategy=None):
    sys.path.insert(0, str(ROOT))
    from main import main

    def offline(*args, **kwargs):
        raise AssertionError('Network access attempted by offline fault probe')

    class Client:
        def __init__(self, config):
            self.calls = 0
            self.by_id = {}
            self.lock = threading.Lock()

        def complete(self, messages, **kwargs):
            query_id = re.search(r'Query ID: ([\w-]+)', messages[1]['content'])[1]
            with self.lock:
                self.calls += 1
                self.by_id[query_id] = self.by_id.get(query_id, 0) + 1
                count = self.by_id[query_id]
            if kind in ('500', '402') or (kind == '429' and count == 1):
                error = RuntimeError(f'HTTP {kind}: insufficient credit' if kind == '402' else f'HTTP {kind}')
                error.status_code = int(kind)
                raise error
            if kind == 'timeout':
                raise TimeoutError('Provider request timed out')
            if kind == 'slow':
                time.sleep(6)
            code = {
                'garbage': 'This is prose, with no Python code block.',
                'loop': '```python\nwhile True:\n    pass\n```',
                # A single contiguous allocation, rejected by the real 700 MiB
                # RLIMIT_AS before it can consume 2 GiB of physical memory.
                'memory': '```python\nanswer = "x" * (2 * 1024 ** 3)\n```',
            }.get(kind, '```python\nanswer = 0\n```')
            return code, 0.0, {'prompt_tokens': 10, 'completion_tokens': 5}

    client = Client(None)
    settings = dict(strategy=strategy or ('vote' if kind == '429' else 'program'),
                    votes=2, vote_mode='fixed', workers=4, program_workers=4,
                    max_attempts=2 if kind == '429' else 1,
                    max_transient_retries=1, transient_retry_sleep=0, rescue_pass=False,
                    program_timeout=1.5, question_timeout=15,
                    total_time_limit=3 if kind == 'slow' else 25,
                    time_guard_margin=.1, fast_path=False, verifier=False,
                    hybrid_text=False)
    args = ['--input', str(data_dir), '--output', str(directory / 'output.csv'), '--profile', 'contest']
    for key, value in settings.items():
        args += ['--set', f'{key}={json.dumps(value)}']
    started = time.monotonic()
    with patch('solver.llm.LLMClient', return_value=client), \
         patch.object(socket.socket, 'connect', offline), \
         patch.object(socket, 'create_connection', offline):
        assert main(args) == 0
    records = [json.loads(line) for path in directory.glob('runs/*/questions.jsonl')
               for line in path.read_text().splitlines()]
    events = [json.loads(line) for path in directory.glob('runs/*/trace.jsonl')
              for line in path.read_text().splitlines()]
    (directory / 'probe.json').write_text(json.dumps(dict(
        calls=client.calls, by_id=client.by_id, seconds=time.monotonic() - started,
        records=records, events=events)))


def run_probe(kind, data_dir, tmp_path, strategy=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('SOLVER_') and key != 'OPENROUTER_BASE_URL'}
    command = [sys.executable, str(Path(__file__).resolve()), kind, str(data_dir), str(tmp_path)]
    if strategy:
        command.append(strategy)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                            text=True, timeout=28)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / 'output.csv'
    assert output.is_file()
    with output.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['id', 'answer']
        rows = list(reader)
    queries = json.loads((data_dir / 'queries.json').read_text(encoding='utf-8-sig'))
    ids = [query['id'] for query in queries]
    assert len(ids) == len(set(ids)) == len(rows) == 20
    assert [row['id'] for row in rows] == ids
    if kind == '429':  # recovered: answers come from the fake model, not fallbacks
        assert all(row['answer'] for row in rows)
    else:
        assert [row['answer'] for row in rows] == FALLBACKS
    probe = json.loads((tmp_path / 'probe.json').read_text())
    assert probe['seconds'] < (5 if kind == 'slow' else 25)
    assert probe['calls'] > 0, 'Fault never reached the provider: ' + result.stdout + result.stderr
    return probe


@pytest.mark.parametrize('kind', FAULTS)
def test_cli_preserves_complete_submission_under_fault(kind, data_dir, tmp_path):
    probe = run_probe(kind, data_dir, tmp_path)
    records = probe['records']
    if kind == '429':
        # The second independent vote recovers; this is separate from the known
        # missing exception retry in the program strategy below.
        assert len(records) == 20
        assert all(record['error'] is None for record in records)
        assert max(probe['by_id'].values()) <= 3
    elif kind == '402':
        assert probe['calls'] <= 4  # At most the already in-flight workers.
        assert any(event['kind'] == 'fatal' for event in probe['events'])
    elif kind == 'slow':
        assert probe['calls'] <= 4
        assert 2.5 <= probe['seconds'] < 5
        assert any(event['kind'] == 'run_finished' for event in probe['events'])
        assert not records  # Guard returns before the sleeping provider does.
    else:
        assert len(records) == 20
        assert all(record['error'] for record in records)
        expected = {'500': 'HTTP 500', 'timeout': 'TimeoutError',
                    'garbage': 'SyntaxError', 'loop': 'TimeoutExpired',
                    'memory': 'MemoryError'}[kind]
        assert all(expected in record['error'] for record in records)


def test_429_program_retry_recovers(data_dir, tmp_path):
    probe = run_probe('429', data_dir, tmp_path, strategy='program')
    # Keep CSV/deadline assertions above mandatory, even for this known bug.
    assert len(probe['records']) == 20
    if all(record['error'] and 'HTTP 429' in record['error'] for record in probe['records']):
        assert probe['calls'] == 20
        pytest.xfail('solver/strategies.py: program breaks on HTTP 429 instead of using remaining attempts')
    assert all(record['error'] is None for record in probe['records'])
    assert max(probe['by_id'].values()) <= 3


if __name__ == '__main__':
    _probe(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]),
           sys.argv[4] if len(sys.argv) > 4 else None)
