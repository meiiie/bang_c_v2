"""Offline durable progress and resume tests with generated workbooks."""
import csv
import errno
import json
import re
import socket
import threading
import time
from collections import Counter

import httpx
import openpyxl
import pytest
from openai import APIConnectionError

from main import main
from solver.checkpoint import answer_config_hash
from solver.config import load_config
from solver.strategies import _transient


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'data'
    (root / 'xlsx_files').mkdir(parents=True)
    workbook = openpyxl.Workbook()
    workbook.active.append(['Name', 'Value'])
    workbook.active.append(['Sample', 42])
    workbook.save(root / 'xlsx_files' / 'small.xlsx')
    return root


def queries(root, count):
    items = [{'id': f'q{i:04}', 'file_name': 'small.xlsx', 'query': 'Return the value as an integer.'}
             for i in range(count)]
    (root / 'queries.json').write_text(json.dumps(items))
    return items


class Client:
    def __init__(self, drop_after=None, error=None):
        self.ids = []
        self.drop_after = drop_after
        self.error = error
        self.lock = threading.Lock()
        self.before_call = None

    def complete(self, messages, **kwargs):
        query_id = re.search(r'Query ID: ([\w-]+)', messages[1]['content'])[1]
        with self.lock:
            self.ids.append(query_id)
            count = len(self.ids)
        if self.before_call:
            self.before_call()
        if self.drop_after is not None and count > self.drop_after:
            raise self.error()
        return 'answer = 42', 0.0, {}


def run(root, tmp_path, monkeypatch, client, extra=(), workers=1):
    monkeypatch.setattr('solver.llm.LLMClient', lambda config: client)
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps({'strategy': 'program', 'max_attempts': 2,
                                  'transient_retry_sleep': 0, 'workers': workers,
                                  'max_transient_retries': 1, 'rescue_pass': False}))
    output = tmp_path / 'result.csv'
    assert main(['--input', str(root), '--output', str(output), '--config', str(config), *extra]) == 0
    with output.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows, tmp_path / 'result.csv.checkpoint.jsonl'


def resume_counts(tmp_path):
    return [json.loads(line)['data']['resumed_ids']
            for path in tmp_path.glob('runs/*/trace.jsonl') for line in path.read_text().splitlines()
            if json.loads(line)['kind'] == 'checkpoint_resume']


def test_scale_500(dataset, tmp_path, monkeypatch):
    items = queries(dataset, 500)
    client = Client()
    started = time.monotonic()
    rows, checkpoint = run(dataset, tmp_path, monkeypatch, client, workers=8)
    assert time.monotonic() - started < 60
    assert [r['id'] for r in rows] == [q['id'] for q in items]
    assert len({r['id'] for r in rows}) == 500
    assert all(r['answer'] == '42' for r in rows)
    records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(records) == len({r['key'] for r in records}) == 500
    assert len(client.ids) == 500


@pytest.mark.parametrize('error', [
    lambda: APIConnectionError(request=httpx.Request('POST', 'https://offline.test')),
    lambda: ConnectionError('offline'),
])
def test_network_drop_and_resume(dataset, tmp_path, monkeypatch, error):
    items = queries(dataset, 6)
    dropping = Client(drop_after=2, error=error)
    rows, checkpoint = run(dataset, tmp_path, monkeypatch, dropping)
    assert [r['id'] for r in rows] == [q['id'] for q in items]
    assert [r['answer'] for r in rows] == ['42'] * 2 + ['0'] * 4
    assert len(checkpoint.read_text().splitlines()) == 2
    assert Counter(dropping.ids) == Counter({q['id']: 1 if i < 2 else 2 for i, q in enumerate(items)})
    healthy = Client()
    # Resumed rows must be in the complete CSV before the first fresh call.
    def check_restored():
        with (tmp_path / 'result.csv').open(newline='') as stream:
            progress = list(csv.DictReader(stream))
        assert len(progress) == 6
        assert [r['answer'] for r in progress[:2]] == ['42', '42']
    healthy.before_call = check_restored
    rows, checkpoint = run(dataset, tmp_path, monkeypatch, healthy)
    assert healthy.ids == [q['id'] for q in items[2:]]
    assert all(r['answer'] == '42' for r in rows)
    assert len(checkpoint.read_text().splitlines()) == 6
    assert sorted(resume_counts(tmp_path)) == [0, 2]


@pytest.mark.parametrize('tail', [b'{"key":', b'bad JSON\n', b'\xff\xfe'])
def test_corrupt_trailing_line(dataset, tmp_path, monkeypatch, tail):
    queries(dataset, 2)
    _, checkpoint = run(dataset, tmp_path, monkeypatch, Client())
    with checkpoint.open('ab') as stream:
        stream.write(tail)
    client = Client()
    rows, _ = run(dataset, tmp_path, monkeypatch, client)
    assert not client.ids and all(r['answer'] == '42' for r in rows)
    # New answers after the corrupt tail also survive the following resume.
    queries(dataset, 3)
    client = Client()
    run(dataset, tmp_path, monkeypatch, client)
    assert client.ids == ['q0002']
    client = Client()
    run(dataset, tmp_path, monkeypatch, client)
    assert not client.ids


def test_changed_query_invalidates(dataset, tmp_path, monkeypatch):
    items = queries(dataset, 2)
    run(dataset, tmp_path, monkeypatch, Client())
    items[0]['query'] = 'What is the sample value? Return an integer.'
    (dataset / 'queries.json').write_text(json.dumps(items))
    client = Client()
    rows, checkpoint = run(dataset, tmp_path, monkeypatch, client)
    assert client.ids == ['q0000']
    assert all(r['answer'] == '42' for r in rows)
    assert len(checkpoint.read_text().splitlines()) == 3


def test_workbook_and_settings_invalidate(dataset, tmp_path, monkeypatch):
    queries(dataset, 1)
    run(dataset, tmp_path, monkeypatch, Client())
    path = dataset / 'xlsx_files' / 'small.xlsx'
    workbook = openpyxl.load_workbook(path)
    workbook.active['B2'] = 43
    workbook.save(path)
    client = Client()
    run(dataset, tmp_path, monkeypatch, client)
    assert client.ids == ['q0000']
    client = Client()
    run(dataset, tmp_path, monkeypatch, client, extra=['--set', 'reasoning=off'])
    assert client.ids == ['q0000']
    config = load_config()
    assert answer_config_hash(config) != answer_config_hash(dict(config, serialization='compact'))
    assert answer_config_hash(config) == answer_config_hash(dict(config, workers=1, checkpoint_path='other',
                                                                max_abandoned_workers=0))


def test_fresh_disabled_and_custom_path(dataset, tmp_path, monkeypatch):
    queries(dataset, 2)
    custom = tmp_path / 'saved' / 'answers.jsonl'
    run(dataset, tmp_path, monkeypatch, Client(), extra=['--set', f'checkpoint_path={custom}'])
    original = custom.read_bytes()
    client = Client()
    run(dataset, tmp_path, monkeypatch, client, extra=['--checkpoint', str(custom)])
    assert not client.ids
    client = Client()
    run(dataset, tmp_path, monkeypatch, client, extra=['--checkpoint', str(custom), '--no-checkpoint', '--fresh'])
    assert len(client.ids) == 2 and custom.read_bytes() == original
    client = Client()
    run(dataset, tmp_path, monkeypatch, client, extra=['--checkpoint', str(custom), '--fresh'])
    assert len(client.ids) == 2 and len(custom.read_text().splitlines()) == 2
    for mode in ('mock', 'dump'):
        run(dataset, tmp_path, monkeypatch, Client(), extra=['--checkpoint', str(custom), '--mode', mode, '--fresh'])
        assert custom.read_bytes() == original


def test_resume_before_provider_setup(dataset, tmp_path, monkeypatch):
    queries(dataset, 1)
    run(dataset, tmp_path, monkeypatch, Client())
    def broken(config):
        raise AssertionError('Fully resumed run should not initialize provider')
    monkeypatch.setattr('solver.llm.LLMClient', broken)
    assert main(['--input', str(dataset), '--output', str(tmp_path / 'result.csv'),
                 '--config', str(tmp_path / 'settings.json')]) == 0
    with (tmp_path / 'result.csv').open(newline='') as stream:
        assert list(csv.DictReader(stream))[0]['answer'] == '42'


@pytest.mark.parametrize('error', [
    ConnectionError('offline'), socket.gaierror(socket.EAI_AGAIN, 'DNS failed'),
    OSError(errno.ENETUNREACH, 'offline'), OSError(errno.EHOSTUNREACH, 'offline'),
    httpx.ConnectError('offline'),
    APIConnectionError(request=httpx.Request('POST', 'https://offline.test')),
])
def test_network_errors_are_transient(error):
    assert _transient(error)


def test_disk_error_is_not_transient():
    assert not _transient(OSError(errno.ENOSPC, 'disk full'))
