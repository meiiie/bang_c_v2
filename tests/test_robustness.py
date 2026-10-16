"""Transport recovery and rescue regressions; all providers are offline doubles."""
import copy
import json
import threading
import time
from types import SimpleNamespace as NS

import httpx
import pytest

from solver.config import load_config, write_csv
from solver.llm import LLMClient, StallError
from solver.pipeline import Pipeline
from solver.strategies import program


def response(text='answer=42'):
    return NS(provider='healthy', usage=NS(cost=.001),
              choices=[NS(message=NS(content=text), finish_reason='stop')])


def transport_factory(monkeypatch, outcomes):
    clients, calls = [], []
    def factory(**settings):
        index = len(clients)
        def create(**kwargs):
            calls.append((index, copy.deepcopy(kwargs)))
            value = outcomes.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        client = NS(chat=NS(completions=NS(create=create)), closed=threading.Event())
        client.close = client.closed.set
        clients.append(client)
        return client
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline')
    monkeypatch.setattr('openai.OpenAI', factory)
    config = dict(load_config(), stream=False, hedge_after=None, transient_retry_sleep=0)
    return config, clients, calls


@pytest.mark.parametrize('error', [httpx.ReadError('SSL RECORD_LAYER_FAILURE'),
                                  httpx.ReadTimeout('read timeout'), StallError('idle')])
def test_transient_failure_uses_fresh_pool(monkeypatch, error):
    config, clients, calls = transport_factory(monkeypatch, [error, response()])
    client = LLMClient(config)
    with pytest.raises(type(error)):
        client.complete([])
    assert client.complete([])[0] == 'answer=42'
    assert [index for index, _ in calls] == [0, 1]
    assert clients[0].closed.wait(.5)
    assert calls[1][1]['extra_body']['provider']['sort'] == 'throughput'


def test_failed_provider_ignored_process_wide_and_expires(monkeypatch):
    class FailedStream:
        def __iter__(self):
            yield NS(provider='bad-upstream', choices=[], usage=None, error=None)
            raise httpx.ReadError('SSL RECORD_LAYER_FAILURE')
        def close(self): pass
    config, _, calls = transport_factory(monkeypatch, [FailedStream(), response(), response()])
    config.update(stream=True, provider_ignore=True, provider_failure_ttl=.05,
                  provider_routing={'sort': 'latency', 'allow_fallbacks': True})
    client = LLMClient(config)
    with pytest.raises(httpx.ReadError): client.complete([])
    other = LLMClient(config)
    other.complete([])
    routing = calls[1][1]['extra_body']['provider']
    assert routing['ignore'] == ['bad-upstream']
    assert routing['allow_fallbacks'] is True
    assert 'ignore' not in client.provider_routing
    time.sleep(.06)
    other.complete([])
    assert 'ignore' not in calls[2][1]['extra_body']['provider']


def test_transient_retries_do_not_consume_execution_attempts(monkeypatch):
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls <= 4: raise httpx.ReadError('SSL RECORD_LAYER_FAILURE')
            return 'answer=42', .001, {}
    config = dict(load_config(), max_attempts=1, transient_retry_sleep=0)
    monkeypatch.setattr('solver.strategies.run_program', lambda *a, **k: {'answer': 42})
    client = Client()
    answer, _, attempts, _ = program(client=client, messages=[], query={'id': 'q', 'query': 'integer'},
        path='unused', config=config, budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer == '42' and len(attempts) == 5


def test_rescue_overwrites_fallback_and_preserves_budget(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    budgets = []
    def strategy(**context):
        budgets.append(context['budget'])
        return (None, .004, [], 'provider failed') if len(budgets) == 1 else ('42', .002, [], None)
    monkeypatch.setitem(__import__('solver.pipeline', fromlist=['STRATEGIES']).STRATEGIES, 'program', strategy)
    config = dict(load_config(), strategy='program', rescue_pass=True)
    pipeline = Pipeline(None, config=config)
    query = {'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}
    # input_hash requires queries.json from the fixture, which is already present.
    output = tmp_path/'submission.csv'
    snapshots = []
    def progress(row):
        snapshots.append(row.copy())
        write_csv(output, [row])
    rows = pipeline.solve_all([query], data_dir, workers=1, output_path=output, on_answer=progress)
    assert rows == [{'id': 'q', 'answer': '42'}]
    assert [r['answer'] for r in snapshots] == ['0', '42']
    assert budgets == pytest.approx([.03, .026])
    assert '42' in output.read_text()
    assert '42' in (tmp_path/'submission.csv.checkpoint.jsonl').read_text()
    records = [json.loads(line) for line in pipeline.log_path.read_text().splitlines()]
    assert sum(r['cost'] for r in records) == pytest.approx(.006)
    assert records[-1]['total_cost'] == pytest.approx(.006)


def test_rescue_respects_time_guard(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def strategy(**context):
        calls.append(context)
        context['stopped'].deadline = time.monotonic()  # main result has no answer
        pipeline.deadline = time.monotonic() + config['time_guard_margin']
        return None, 0, [], 'deadline'
    monkeypatch.setitem(__import__('solver.pipeline', fromlist=['STRATEGIES']).STRATEGIES, 'program', strategy)
    config = dict(load_config(), strategy='program', rescue_pass=True)
    pipeline = Pipeline(None, config=config)
    query = {'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}
    rows = pipeline.solve_all([query], data_dir, workers=1)
    assert rows == [{'id': 'q', 'answer': '0'}]
    assert len(calls) == 1


def test_transient_retry_cap(monkeypatch):
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise httpx.ReadError('offline')
    config = dict(load_config(), max_transient_retries=8, transient_retry_sleep=0)
    client = Client()
    answer, _, attempts, _ = program(client=client, messages=[], query={'id': 'q', 'query': 'integer'},
        path='unused', config=config, budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer is None and client.calls == len(attempts) == 9


def test_pool_retirement_preserves_other_active_request(monkeypatch):
    release, entered = threading.Event(), threading.Event()
    config, clients, calls = transport_factory(monkeypatch, [])
    client = LLMClient(config)
    def create(**kwargs):
        if threading.current_thread().name == 'llm-request' and not entered.is_set():
            entered.set()
            assert release.wait(2)
            return response('other')
        raise httpx.ReadError('SSL RECORD_LAYER_FAILURE')
    clients[0].chat.completions.create = create
    from solver.config import launch
    future = launch(client.complete, [])
    assert entered.wait(1)
    try:
        with pytest.raises(httpx.ReadError): client.complete([])
        assert client.client is not clients[0]
        assert not clients[0].closed.is_set()
    finally:
        release.set()
    assert future.result(timeout=2)[0] == 'other'
    assert clients[0].closed.wait(.5)


@pytest.mark.parametrize('key,value', [('max_transient_retries', True), ('max_transient_retries', -1),
    ('provider_failure_ttl', 0), ('provider_failure_ttl', float('inf')),
    ('provider_ignore', 1), ('provider_ignore', 'false'), ('provider_ignore', None),
    ('fresh_client_on_transient', 1), ('rotate_provider_sort', 1), ('rescue_pass', 1)])
def test_robustness_config_validation(key, value):
    from solver.config import validate_config
    with pytest.raises(ValueError, match=key):
        validate_config(dict(load_config(), **{key: value}))


def test_rescue_request_cannot_overrun_global_guard(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def strategy(**context):
        calls.append(context)
        if len(calls) > 1:
            entered.set()
            release.wait(2)
        return None, .001, [], 'provider failed'
    monkeypatch.setitem(__import__('solver.pipeline', fromlist=['STRATEGIES']).STRATEGIES, 'program', strategy)
    config = dict(load_config(), strategy='program', rescue_pass=True, total_time_limit=.2,
                  time_guard_margin=.02, question_timeout=10)
    pipeline = Pipeline(None, config=config)
    started = time.monotonic()
    try:
        rows = pipeline.solve_all([{'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}], data_dir)
        assert entered.is_set() and len(calls) == 2
        assert rows[0]['answer'] == '0'
        assert time.monotonic()-started < .5
        assert calls[1]['stopped'].deadline == pipeline.deadline-.02
    finally:
        release.set()


def test_rescue_skips_exhausted_budget(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def strategy(**context):
        calls.append(context)
        return None, context['budget'], [], 'budget exhausted'
    monkeypatch.setitem(__import__('solver.pipeline', fromlist=['STRATEGIES']).STRATEGIES, 'program', strategy)
    pipeline = Pipeline(None, config=dict(load_config(), strategy='program', rescue_pass=True))
    pipeline.solve_all([{'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}], data_dir)
    assert len(calls) == 1


@pytest.mark.parametrize('initial,expected', [('latency', ['latency', 'throughput', 'latency']),
    ('throughput', ['throughput', 'latency', 'throughput'])])
def test_retry_sort_alternates_and_does_not_mutate_config(monkeypatch, initial, expected):
    config, _, calls = transport_factory(monkeypatch, [httpx.ReadError('offline'),
                                                       StallError('idle'), response()])
    config['provider_routing'] = {'sort': initial, 'allow_fallbacks': True}
    client = LLMClient(config)
    for _ in range(2):
        with pytest.raises((httpx.ReadError, StallError)): client.complete([])
    client.complete([])
    assert [kw['extra_body']['provider']['sort'] for _, kw in calls] == expected
    assert client.provider_routing == config['provider_routing']


def test_watchdog_excludes_provider_from_last_chunk(monkeypatch):
    from solver import llm
    from tests.test_streaming import Stream, chunk
    delta = chunk('partial')
    delta.provider = 'watchdog-upstream'
    stream = Stream([(0, delta), (2, chunk('tail'))])
    config, _, calls = transport_factory(monkeypatch, [stream, response()])
    config.update(stream=True, provider_ignore=True, stall_timeout=.02, provider_failure_ttl=.1)
    client = LLMClient(config)
    try:
        with pytest.raises(StallError): client.complete([])
        client.complete([])
        assert 'watchdog-upstream' in calls[1][1]['extra_body']['provider']['ignore']
    finally:
        with llm._provider_lock:
            llm._provider_failures.pop('watchdog-upstream', None)


def test_rescue_publishes_completed_abandoned_success(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first_finished = threading.Event()
    def strategy(**context):
        if context['query']['id'] == 'late':
            time.sleep(.04)
            first_finished.set()
        else:
            assert first_finished.wait(1)
        return '42', .001, [], None
    monkeypatch.setitem(__import__('solver.pipeline', fromlist=['STRATEGIES']).STRATEGIES, 'program', strategy)
    config = dict(load_config(), strategy='program', question_timeout=.03, total_time_limit=2,
                  time_guard_margin=.01, rescue_pass=True)
    pipeline = Pipeline(None, config=config)
    queries = [{'id': q, 'file_name': 'p4_t01.xlsx', 'query': 'integer'} for q in ('late', 'next')]
    snapshots = []
    rows = pipeline.solve_all(queries, data_dir, workers=1, output_path=tmp_path/'out.csv',
                              on_answer=lambda r: snapshots.append(r.copy()))
    assert all(r['answer'] == '42' for r in rows)
    assert {'id': 'late', 'answer': '42'} in snapshots
    saved = [json.loads(line) for line in (tmp_path/'out.csv.checkpoint.jsonl').read_text().splitlines()]
    assert {r['id'] for r in saved} == {'late', 'next'}
    assert pipeline.question_records['late']['total_cost'] == pytest.approx(.001)


@pytest.fixture
def failed_registry(monkeypatch):
    from solver import llm
    registry = {'upstream-a': time.monotonic() + 120, 'upstream-b': time.monotonic() + 120}
    monkeypatch.setattr(llm, '_provider_failures', registry)
    return registry


@pytest.mark.parametrize('profile', ['contest', 'dev-glm'])
def test_provider_ignore_disabled_by_default(profile):
    assert load_config(profile=profile)['provider_ignore'] is False


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('manual', [[], ['manual-a', 'manual-b']])
def test_provider_ignore_filters_are_disabled_or_capped(monkeypatch, failed_registry, enabled, manual):
    config, clients, calls = transport_factory(monkeypatch, [httpx.ReadError('offline'), response()])
    config.update(provider_ignore=enabled,
                  provider_routing={'sort': 'latency', 'ignore': manual})
    client = LLMClient(config)
    with pytest.raises(httpx.ReadError): client.complete([])
    assert client.complete([])[0] == 'answer=42'
    for _, kwargs in calls:
        routing = kwargs['extra_body']['provider']
        if enabled:
            assert routing['ignore'] == (['manual-a'] if manual else ['upstream-a'])
        else:
            assert 'ignore' not in routing
    assert calls[1][0] == 1
    assert calls[1][1]['extra_body']['provider']['sort'] == 'throughput'
    assert client.provider_routing['ignore'] == manual


@pytest.mark.parametrize('status', [400, 404])
@pytest.mark.parametrize('stream', [False, True])
def test_ignored_providers_retry_without_filter(monkeypatch, failed_registry, status, stream):
    from solver.llm import is_transient
    from tests.test_streaming import Stream, chunk
    error = RuntimeError('All providers have been ignored; Filter by Ignored Providers')
    error.status_code = status
    failure = Stream([(0, chunk(error={'code': status, 'message': str(error)}))]) if stream else error
    config, _, calls = transport_factory(monkeypatch, [failure, response(), response()])
    config.update(provider_ignore=True, stream=stream,
                  provider_routing={'sort': 'latency', 'ignore': ['manual']})
    client = LLMClient(config)
    assert client.complete([])[0] == 'answer=42'
    assert len(calls) == 2
    assert calls[0][1]['extra_body']['provider']['ignore'] == ['manual']
    assert 'ignore' not in calls[1][1]['extra_body']['provider']
    assert failed_registry == {}
    if stream:
        assert calls[1][1]['timeout'].as_dict().keys() == calls[0][1]['timeout'].as_dict().keys()
        for operation, timeout in calls[1][1]['timeout'].as_dict().items():
            assert timeout <= calls[0][1]['timeout'].as_dict()[operation]
    else:
        assert calls[1][1]['timeout'] <= calls[0][1]['timeout']
    assert is_transient(error)
    client.provider_routing.pop('ignore')
    client.complete([])
    assert 'ignore' not in calls[2][1]['extra_body']['provider']


@pytest.mark.parametrize('status,message,retry', [
    (404, 'All providers have been ignored', True),
    (400, 'Ignored providers exclude all endpoints', True),
    (404, 'Model not found', False), (400, 'Invalid request', False),
    (403, 'All providers have been ignored', False),
])
def test_ignore_recovery_is_bounded_and_specific(monkeypatch, failed_registry, status, message, retry):
    from solver.llm import is_transient
    error = RuntimeError(message)
    error.status_code = status
    config, _, calls = transport_factory(monkeypatch, [error, error])
    config['provider_ignore'] = True
    client = LLMClient(config)
    with pytest.raises(RuntimeError, match=message): client.complete([])
    assert len(calls) == (2 if retry else 1)
    assert is_transient(error) is retry
    if retry:
        assert 'ignore' not in calls[1][1]['extra_body']['provider']
        assert failed_registry == {}
    else: assert len(failed_registry) == 2


def test_ignore_recovery_discards_partial_stream(monkeypatch, failed_registry):
    from tests.test_streaming import Stream, chunk, good
    rejected = Stream([(0, chunk('stale')),
                       (0, chunk(error={'code': 404, 'message': 'All providers have been ignored'}))])
    config, _, calls = transport_factory(monkeypatch, [rejected, good('answer=42')])
    config.update(provider_ignore=True, stream=True)
    client = LLMClient(config)
    assert client.complete([])[0] == 'answer=42'
    assert len(calls) == 2
