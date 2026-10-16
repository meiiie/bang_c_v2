import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import zipfile

import pytest
from main import main
from solver.config import load_config, apply_overrides, fallback_answer, validate_config
from solver.llm import LLMClient
from solver.pipeline import Pipeline


@pytest.mark.parametrize('kind,level,expected', [
    ('openrouter', 'off', {'usage': {'include': True}, 'reasoning': {'enabled': False}}),
    ('openrouter', 'medium', {'usage': {'include': True}, 'reasoning': {'effort': 'medium'}}),
    ('zai', 'off', {'thinking': {'type': 'disabled'}}),
    ('zai', 'low', {'thinking': {'type': 'enabled'}}),
    ('zai', 'medium', {'thinking': {'type': 'enabled'}}),
    ('openai_compatible', 'low', {}),
])
def test_provider_body_and_cost(monkeypatch, kind, level, expected):
    config = load_config(profile='dev-glm')
    config.update(provider_kind=kind, reasoning=level, input_price=2, output_price=4)
    if kind == 'openrouter': config['model'] = 'qwen/qwen3.7-flash'
    monkeypatch.setenv('ZAI_API_KEY', 'offline')
    client = LLMClient(config)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='answer=1'), finish_reason='stop')])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete([{'content': 'test'}])[1] == pytest.approx(.0001)
    assert calls[0]['extra_body'] == expected


def test_profile_selection(monkeypatch):
    assert load_config()['profile'] == 'contest'
    monkeypatch.setenv('SOLVER_PROFILE', 'dev-glm')
    config = load_config()
    assert config['model'] == 'glm-5.3'
    assert config['provider_kind'] == 'zai' and config['api_key_env'] == 'ZAI_API_KEY'
    assert config['base_url'] == 'https://api.z.ai/api/coding/paas/v4'
    assert config['input_price'] == config['output_price'] == 0
    assert load_config(profile='contest')['provider_kind'] == 'openrouter'
    assert apply_overrides(config, ['model=other', 'base_url=https://local.test/v1'])['model'] == 'other'
    with pytest.raises(ValueError): apply_overrides(config, ['provider_kind=openrouter'])


def check_csv(path, queries):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['id', 'answer']
        rows = list(reader)
    assert [row['id'] for row in rows] == [q['id'] for q in queries]
    assert len({row['id'] for row in rows}) == len(queries)
    return rows


@pytest.mark.parametrize('status', [401, 402, 429, 500, 503])
def test_main_always_valid_when_calls_raise(data_dir, tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    queries = json.loads((data_dir/'queries.json').read_text())
    output = tmp_path/'submission.csv'
    class Client:
        def __init__(self, config): pass
        def complete(self, *args, **kwargs):
            # The entire CSV must already exist at the first API call.
            check_csv(output, queries)
            error = RuntimeError(f'HTTP {status}')
            error.status_code = status
            raise error
    monkeypatch.setattr('solver.llm.LLMClient', Client)
    main(['--input', str(data_dir), '--output', str(output), '--workers', '2',
          '--set', 'transient_retry_sleep=0'])
    rows = check_csv(output, queries)
    assert [r['answer'] for r in rows] == [fallback_answer(q) for q in queries]


@pytest.mark.parametrize('failure', ['key', 'config', 'parse', 'crash'])
def test_initialization_and_parse_failures(data_dir, tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    def broken(*args, **kwargs): raise RuntimeError('failure')
    extra = []
    if failure == 'config': extra = ['--set', 'model=invalid']
    if failure == 'parse':
        monkeypatch.setattr('solver.pipeline.parse_workbook', broken)
        extra = ['--mode', 'dump']
    if failure == 'crash':
        monkeypatch.setattr('solver.pipeline.Pipeline.solve_all', broken)
        extra = ['--mode', 'mock']
    output = tmp_path/'out.csv'
    main(['--input', str(data_dir), '--output', str(output), *extra])
    check_csv(output, json.loads((data_dir/'queries.json').read_text()))


def test_time_guard_stops_calls(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            return 'answer=1/0', 0
    config = load_config()
    config.update(strategy='program', total_time_limit=.05, time_guard_margin=.05, workers=1)
    client = Client()
    pipeline = Pipeline(client, config=config)
    queries = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for i in range(3)]
    assert len(pipeline.solve_all(queries, data_dir)) == 3
    assert client.calls == 0


def test_question_wall_deadline_and_no_thread_explosion(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            release.wait(10)
            return 'answer=1', 0
    config = load_config()
    config.update(strategy='program', question_timeout=.15, workers=1, max_abandoned_workers=0)
    client = Client()
    pipeline = Pipeline(client, config=config)
    queries = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for i in range(5)]
    started = time.monotonic()
    try:
        assert all(r['answer'] == '0' for r in pipeline.solve_all(queries, data_dir))
        assert time.monotonic()-started < 1
        assert client.calls == 1
    finally:
        release.set()


def test_question_timeouts_replace_workers(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    lock = threading.Lock()
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            with lock:
                self.calls += 1
                call = self.calls
            if call <= 2:
                release.wait(10)
            return 'answer=42', 0
    config = apply_overrides(load_config(), ['strategy=program', 'workers=2', 'question_timeout=2'])
    events = []
    pipeline = Pipeline(Client(), config=config, on_event=lambda kind, data: events.append((kind, data)))
    queries = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for i in range(7)]
    try:
        rows = pipeline.solve_all(queries, data_dir)
        assert [row['answer'] for row in rows] == ['0', '0', '42', '42', '42', '42', '42']
        assert sum(kind == 'worker_replaced' for kind, _ in events) == 2
    finally:
        release.set()


@pytest.mark.parametrize('cap', [0, 1, 2])
def test_abandoned_worker_cap_stops_scheduling(data_dir, tmp_path, monkeypatch, cap):
    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            release.wait(10)
            return 'answer=42', 0
    config = load_config()
    config.update(strategy='program', workers=1, question_timeout=.15, max_abandoned_workers=cap)
    events = []
    client = Client()
    pipeline = Pipeline(client, config=config, on_event=lambda kind, data: events.append((kind, data)))
    queries = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for i in range(6)]
    try:
        assert [row['answer'] for row in pipeline.solve_all(queries, data_dir)] == ['0'] * 6
        assert client.calls == cap + 1
        assert sum(kind == 'worker_replaced' for kind, _ in events) == cap
    finally:
        release.set()


@pytest.mark.parametrize('value', ['-1', '1.5', 'true', '"2"', 'null'])
def test_max_abandoned_workers_rejects_invalid_values(value):
    config = load_config()
    config['max_abandoned_workers'] = json.loads(value)
    with pytest.raises(ValueError, match='max_abandoned_workers'):
        validate_config(config)


def test_atomic_progress_survives_crash(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path/'out.csv'
    queries = json.loads((data_dir/'queries.json').read_text())
    def crash(self, queries, *args, on_answer, **kwargs):
        on_answer({'id': queries[0]['id'], 'answer': 'saved'})
        assert check_csv(output, queries)[0]['answer'] == 'saved'
        raise RuntimeError('crash after first answer')
    monkeypatch.setattr(Pipeline, 'solve_all', crash)
    main(['--input', str(data_dir), '--output', str(output), '--mode', 'mock'])
    assert check_csv(output, queries)[0]['answer'] == 'saved'
    assert not list(tmp_path.glob('*.tmp'))


def test_build_zip_tracked_only(tmp_path):
    spec = importlib.util.spec_from_file_location('build_zip', Path(__file__).resolve().parents[1]/'tools/build_zip.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    names = ['README.md', 'main.py', 'eval.py', 'requirements.txt', 'configs/default.json',
             'solver/config.py', 'tests/data.json', 'tools/dev.py', 'runs/result.json', 'prompts/q.txt', 'robust/q.json']
    for name in set(names) | module.REQUIRED | {'solver/untracked.py'}:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('example')
    subprocess.run(['git', 'add', '--', *names], cwd=tmp_path, check=True)
    destination = module.build(tmp_path)
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == module.REQUIRED | {'solver/config.py'}
    assert destination.stat().st_size < 100*1024**2


def test_provider_config_file_overrides_profile(tmp_path):
    path = tmp_path/'provider.json'
    path.write_text(json.dumps({'profile': 'dev-glm', 'base_url': 'https://custom.test/v1',
                                'model': 'custom-model', 'api_key_env': 'CUSTOM_KEY', 'input_price': .5}))
    config = load_config(path)
    assert config['base_url'] == 'https://custom.test/v1'
    assert config['model'] == 'custom-model' and config['api_key_env'] == 'CUSTOM_KEY'
    assert config['input_price'] == .5


def test_global_deadline_after_first_call(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            release.wait(10)
            return 'answer=1', 0
    config = load_config()
    config.update(strategy='program', workers=1, total_time_limit=.15, time_guard_margin=.01)
    client = Client()
    pipeline = Pipeline(client, config=config)
    queries = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for i in range(3)]
    started = time.monotonic()
    try:
        assert all(r['answer'] == '0' for r in pipeline.solve_all(queries, data_dir))
        assert time.monotonic()-started < 1
        assert client.calls == 1
    finally:
        release.set()


def test_reasoning_only_length_is_error(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline')
    config = load_config()
    config.update(strategy='program', max_attempts=1)
    client = LLMClient(config)
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='', reasoning_content='answer=99'),
                                                        finish_reason='length')])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    pipeline = Pipeline(client, config=config)
    assert pipeline.solve_all([{'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}], data_dir)[0]['answer'] == '0'
    assert 'finish_reason=length' in pipeline.log_path.read_text()


@pytest.mark.parametrize('kind', ['openrouter', 'zai'])
@pytest.mark.parametrize('fallback', ['off', 'low', None])
@pytest.mark.parametrize('first_code', ['', 'Still working on the calculation',
                                        '```python\nanswer=1/0\n```'])
def test_length_fallback_is_per_sample(monkeypatch, kind, fallback, first_code):
    from solver.strategies import program
    config = apply_overrides(load_config(), [f'length_fallback_reasoning={json.dumps(fallback)}'])
    config.update(reasoning='medium')
    if kind == 'zai':
        config.update(profile='dev-glm', provider_kind=kind, api_key_env='ZAI_API_KEY')
    monkeypatch.setenv(config['api_key_env'], 'offline')
    client = LLMClient(config)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        attempt = (len(calls) - 1) % 3
        content = [first_code, 'answer=1/0', '```python\nanswer=42\n```'][attempt]
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4000, cost=.001,
                                  completion_tokens_details=SimpleNamespace(reasoning_tokens=3990)),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content='answer=99'),
                                     finish_reason='length' if attempt == 0 else 'stop')])
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('solver.strategies.run_program',
                        lambda code, *args, **kwargs: {'answer': 42} if 'answer=42' in code else {'error': 'Invalid program'})
    original_body = client.extra_body()
    fallback_body = client.extra_body(fallback)
    for sample_id in range(2):
        answer, cost, attempts, error = program(
            client=client, messages=[{'role': 'user', 'content': 'test'}],
            query={'id': 'q', 'query': 'integer'}, path='unused.xlsx', config=config,
            budget=.03, stopped=threading.Event(), on_event=lambda *args: None, sample_id=sample_id)
        assert answer == '42' and error is None
        assert cost == pytest.approx(.003)
        assert len(attempts) == 3
        assert attempts[0]['usage']['finish_reason'] == 'length'
        assert attempts[0]['usage']['reasoning_tokens'] == 3990
    assert [call['extra_body'] for call in calls] == [original_body, fallback_body, fallback_body] * 2
    assert client.reasoning == 'medium'
    assert client.extra_body() == original_body


def test_non_length_failure_keeps_reasoning(monkeypatch):
    from solver.strategies import program
    config = load_config()
    calls = []
    class Client:
        def complete(self, messages, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return '', 0, {'finish_reason': 'stop'}
            return 'answer=42', 0, {'finish_reason': 'stop'}
    monkeypatch.setattr('solver.strategies.run_program', lambda *args, **kwargs: {'answer': 42})
    answer, _, attempts, error = program(
        client=Client(), messages=[], query={'id': 'q', 'query': 'integer'}, path='unused.xlsx',
        config=config, budget=.03, stopped=threading.Event(), on_event=lambda *args: None)
    assert answer == '42' and error is None and len(attempts) == 2
    assert all('reasoning' not in call for call in calls)


@pytest.mark.parametrize('value', ['high', '', 0, True, []])
def test_length_fallback_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='length_fallback_reasoning'):
        apply_overrides(load_config(), [f'length_fallback_reasoning={json.dumps(value)}'])


def test_build_zip_fresh_export(tmp_path):
    """An export with no tracked runtime files must still be installable."""
    spec = importlib.util.spec_from_file_location(
        'build_zip_export', Path(__file__).resolve().parents[1]/'tools/build_zip.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    runtime = {'solver/__init__.py', 'solver/parser.py'}
    excluded = {'solver/__pycache__/parser.pyc', 'solver/answers.csv',
                'data/queries.json', 'inputs/secret.xlsx', 'tests/oracle_programs/q.py'}
    for name in module.REQUIRED | runtime | excluded:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic placeholder')
    with zipfile.ZipFile(module.build(tmp_path)) as archive:
        assert set(archive.namelist()) == module.REQUIRED | runtime
