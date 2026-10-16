"""Exercise evaluation reporting independently of the solver."""
import csv
import json
from pathlib import Path
import subprocess
import sys


EVAL = Path(__file__).resolve().parents[1] / 'eval.py'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'answer'])
        writer.writerows(rows)


def test_type_accuracy_and_missing_predictions(tmp_path):
    gold = tmp_path / 'gold.csv'
    pred = tmp_path / 'pred.csv'
    write_csv(gold, [('a', 'synthetic alpha'), ('b', 'synthetic beta'), ('c', 'synthetic delta'), ('d', 'synthetic epsilon')])
    write_csv(pred, [('a', ' synthetic alpha '), ('b', 'synthetic gamma'), ('extra', '0')])
    (tmp_path / 'queries.json').write_text(json.dumps([
        {'id': 'a', 'type': 'sum'}, {'id': 'b', 'type': 'sum'},
        {'id': 'c', 'type': 'compare'}, {'id': 'd'},
    ]), encoding='utf-8')
    result = subprocess.run(
        [sys.executable, str(EVAL), '--pred', str(pred), '--gold', str(gold)],
        text=True, capture_output=True, check=True,
    )
    assert 'Accuracy: 1/4 (25.00%)' in result.stdout
    assert 'Type sum: 1/2 (50.00%)' in result.stdout
    assert 'Type compare: 0/1 (0.00%)' in result.stdout
    assert 'Type untyped: 0/1 (0.00%)' in result.stdout
    assert 'Unexpected ID: extra' in result.stdout


def test_legacy_and_explicit_queries_path(tmp_path):
    gold = tmp_path / 'gold.csv'
    write_csv(gold, [('a', 'synthetic alpha')])
    command = [sys.executable, str(EVAL), '--pred', str(gold), '--gold', str(gold)]
    legacy = subprocess.run(command, text=True, capture_output=True, check=True)
    assert 'Accuracy: 1/1 (100.00%)' in legacy.stdout
    assert 'Type ' not in legacy.stdout
    queries = tmp_path / 'custom.json'
    queries.write_text(json.dumps([{'id': 'a', 'type': 'lookup'}]), encoding='utf-8')
    typed = subprocess.run(command + ['--queries', str(queries)],
                           text=True, capture_output=True, check=True)
    assert 'Type lookup: 1/1 (100.00%)' in typed.stdout
