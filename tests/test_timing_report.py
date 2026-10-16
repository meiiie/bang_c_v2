import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/timing_report.py'


def test_timing_cli_sums_attempts_and_projects_sequential_guard(tmp_path):
    path = tmp_path / 'questions.jsonl'
    records = [dict(id='q1', seconds=10, samples_used=2, attempts=[
        {'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'reasoning_tokens': 10}},
        {'usage': {'prompt_tokens': 50, 'completion_tokens': 30}}]),
        dict(id='q2', seconds=20, samples_used=0, attempts=[]),
        dict(id='q3', seconds=30, samples_used=1, attempts=[{'usage': {}}])]
    path.write_text('\n'.join(json.dumps(record) for record in records))
    result = subprocess.run([sys.executable, str(SCRIPT), str(path), '--json'],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report['sequential_seconds'] == 60
    assert report['mean_seconds'] == report['p50_seconds'] == 20
    assert report['p90_seconds'] == pytest.approx(28)
    assert report['max_seconds'] == 30
    assert report['samples_used_distribution'] == {'0': 1, '1': 1, '2': 1}
    assert report['tokens_per_question'] == pytest.approx(200 / 3)
    assert report['usage_attempts_missing'] == 1
    assert report['projected_20_seconds'] == 400
    assert report['guard_seconds'] == 2100
    assert report['guard_headroom_seconds'] == 1700
    text = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True,
                          text=True, check=True).stdout
    assert '35-min' in text and '400.00' in text


@pytest.mark.parametrize('content', ['', '{broken', '{"seconds": -1, "samples_used": 0}',
                                     '{"seconds": NaN, "samples_used": 0}'])
def test_timing_cli_rejects_empty_or_invalid_log(tmp_path, content):
    path = tmp_path / 'questions.jsonl'
    path.write_text(content)
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'error:' in result.stderr
    assert 'Traceback' not in result.stderr
