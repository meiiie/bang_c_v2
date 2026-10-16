"""Offline transport doubles exercise watchdogs, accounting and deadline propagation."""
from collections import deque
from types import SimpleNamespace as NS
import threading
import time

import pytest
from solver import llm
from solver.config import DeadlineStop, GuardedClient, load_config, validate_config
from solver.strategies import program


def chunk(content=None, reasoning=None, usage=None, finish=None, error=None):
    return NS(choices=[NS(delta=NS(content=content, reasoning=reasoning),
                                          finish_reason=finish)], usage=usage, error=error)


class Stream:
    def __init__(self, steps):
        self.steps = steps
        self.closed = threading.Event()

    def __iter__(self):
        for delay, value in self.steps:
            if self.closed.wait(delay):
                return
            if isinstance(value, Exception):
                raise value
            yield value

    def close(self):
        self.closed.set()


def good(text='answer=42', cost=.001):
    return Stream([(0, chunk(text, finish='stop')),
                   (0, chunk(usage=NS(cost=cost, prompt_tokens=10, completion_tokens=5,
                                     completion_tokens_details=NS(reasoning_tokens=2))))])


def make_client(monkeypatch, responses, **settings):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline')
    config = load_config()
    config.update(stream=True, first_token_timeout=.08, stall_timeout=.08,
                  hedge_after=None, max_hedges=1, transient_retry_sleep=0)
    config.update(settings)
    queue, calls, lock = deque(responses), [], threading.Lock()
    def create(**kwargs):
        with lock:
            calls.append(kwargs)
            response = queue.popleft()
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response()
        return response
    monkeypatch.setattr('openai.OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    client = llm.LLMClient(config)
    return client, config, calls


MESSAGES = [{'role': 'user', 'content': 'test'}]


def test_midstream_stall_retried_and_charged(monkeypatch):
    stuck = Stream([(0, chunk('answer=')), (2, chunk('1'))])
    client, config, calls = make_client(monkeypatch, [stuck, good()])
    monkeypatch.setattr('solver.strategies.run_program', lambda *a, **k: {'answer': 42})
    answer, cost, attempts, error = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer == '42' and error is None
    assert len(calls) == len(attempts) == 2
    assert stuck.closed.wait(.3)
    assert attempts[0]['cost'] > 0
    assert cost == pytest.approx(client.total_cost)
    assert cost > .001


def test_slow_first_token_watchdog(monkeypatch):
    stuck = Stream([(0, chunk()), (2, chunk('answer=42'))])
    client, _, _ = make_client(monkeypatch, [stuck])
    started = time.monotonic()
    with pytest.raises(llm.StallError, match='first token'):
        client.complete(MESSAGES)
    assert .05 <= time.monotonic() - started < .5
    assert stuck.closed.wait(.3)


def test_reasoning_counts_as_first_token(monkeypatch):
    stream = Stream([(.04, chunk(reasoning='thinking')), (.05, chunk('answer=42')),
                     (0, chunk(usage=NS(cost=.002)))])
    client, _, calls = make_client(monkeypatch, [stream])
    text, cost, meta = client.complete(MESSAGES)
    assert text == 'answer=42' and cost == .002
    assert calls[0]['stream'] is True
    assert calls[0]['stream_options'] == {'include_usage': True}
    assert stream.closed.wait(.3)


@pytest.mark.parametrize('field', ['content', 'reasoning', 'reasoning_content', 'reasoning_details'])
def test_progressing_stream_outlives_request_timeout(monkeypatch, field):
    progress = chunk()
    setattr(progress.choices[0].delta, field,
            [{'text': 'thinking'}] if field == 'reasoning_details' else 'working')
    stream = Stream([(.025, progress)] * 8 + [(0, chunk('answer=42', finish='stop'))])
    client, _, calls = make_client(monkeypatch, [stream], request_timeout=.04,
                                  first_token_timeout=.1, stall_timeout=.1,
                                  stream_max_seconds=None)
    started = time.monotonic()
    assert client.complete(MESSAGES)[0].endswith('answer=42')
    assert time.monotonic() - started >= .2
    assert len(calls) == 1
    # SDK timeouts bound individual reads, independently of the total duration.
    assert calls[0]['timeout'].read == pytest.approx(.1)
    assert stream.closed.wait(.3)


def test_progressing_stream_still_stops_at_stall_timeout(monkeypatch):
    stream = Stream([(0, chunk(reasoning='thinking')), (2, chunk('answer=42'))])
    client, _, _ = make_client(monkeypatch, [stream], request_timeout=.02,
                              stall_timeout=.08)
    started = time.monotonic()
    with pytest.raises(llm.StallError, match='between chunks'):
        client.complete(MESSAGES)
    assert .06 <= time.monotonic() - started < .5
    assert stream.closed.wait(.3)


@pytest.mark.parametrize('cap', ['question', 'stream'])
def test_progressing_stream_obeys_overall_deadline(monkeypatch, cap):
    stream = Stream([(.02, chunk(reasoning='thinking'))] * 50)
    settings = {'stream_max_seconds': .12} if cap == 'stream' else {}
    client, _, _ = make_client(monkeypatch, [stream], request_timeout=.03, **settings)
    stopped = DeadlineStop(threading.Event(), time.monotonic() + .12)
    started = time.monotonic()
    with pytest.raises(llm.StallError, match='deadline'):
        (GuardedClient(client, stopped) if cap == 'question' else client).complete(MESSAGES)
    assert .09 <= time.monotonic() - started < .5
    assert stream.closed.wait(.3)


def test_nonstreaming_keeps_request_timeout(monkeypatch):
    def delayed_response():
        time.sleep(.15)
        return NS(usage=None, choices=[NS(message=NS(content='answer=42'), finish_reason='stop')])
    client, _, calls = make_client(monkeypatch, [delayed_response], stream=False,
                                  request_timeout=.04)
    started = time.monotonic()
    with pytest.raises(llm.StallError, match='deadline'):
        client.complete(MESSAGES, remaining_time=1)
    assert .03 <= time.monotonic() - started < .12
    assert 0 < calls[0]['timeout'] <= .04


def test_first_token_window_is_independent_of_stall_timeout(monkeypatch):
    stream = Stream([(.12, chunk('answer=42', finish='stop'))])
    client, _, _ = make_client(monkeypatch, [stream], request_timeout=.03,
                              first_token_timeout=.2, stall_timeout=.05)
    assert client.complete(MESSAGES)[0] == 'answer=42'


@pytest.mark.parametrize('in_stream', [False, True])
def test_provider_5xx_retries(monkeypatch, in_stream):
    error = RuntimeError('upstream failed')
    error.status_code = 503
    failure = Stream([(0, chunk(error={'code': 503, 'message': 'upstream failed'}))]) if in_stream else error
    client, config, calls = make_client(monkeypatch, [failure, good()])
    monkeypatch.setattr('solver.strategies.run_program', lambda *a, **k: {'answer': 42})
    answer, _, attempts, _ = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer == '42' and len(calls) == len(attempts) == 2


@pytest.mark.parametrize('usage', [None, NS(cost=.003)])
def test_hedge_fast_duplicate_counts_both_and_closes_loser(monkeypatch, usage):
    primary = Stream([(0, chunk(reasoning='thinking', usage=usage)), (2, chunk('slow'))])
    client, _, calls = make_client(monkeypatch, [primary, good('fast', .002)])
    client.hedge_after = .03
    started = time.monotonic()
    text, cost, meta = client.complete(MESSAGES, max_cost=.03)
    assert text == 'fast' and time.monotonic() - started < .3
    assert len(calls) == 2 and primary.closed.wait(.3)
    # 4 conservative input tokens, up to 4000 output tokens at .03/.13 per million.
    expected_loser = .003 if usage else .00052012
    assert cost == pytest.approx(.002 + expected_loser)
    assert client.total_cost == pytest.approx(cost)


@pytest.mark.parametrize('limit', ['budget', 'time', 'disabled'])
def test_hedge_not_launched_without_resources(monkeypatch, limit):
    stream = Stream([(.05, chunk('answer=42')), (0, chunk(usage=NS(cost=.00001)))])
    client, _, calls = make_client(monkeypatch, [stream])
    client.hedge_after = None if limit == 'disabled' else .02
    kwargs = {'max_cost': .000009} if limit == 'budget' else {}
    if limit == 'time': kwargs['remaining_time'] = .06
    assert client.complete(MESSAGES, **kwargs)[0] == 'answer=42'
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['openrouter', 'zai', 'openai_compatible'])
def test_provider_routing_only_openrouter(monkeypatch, kind):
    config = load_config(profile='dev-glm')
    config.update(provider_kind=kind, model='qwen/qwen3.7-flash', stream=False,
                  provider_routing={'sort': 'latency', 'allow_fallbacks': True})
    monkeypatch.setenv('ZAI_API_KEY', 'offline')
    client = llm.LLMClient(config)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(usage=NS(cost=.001), choices=[NS(message=NS(content='answer=1'), finish_reason='stop')])
    client.client = NS(chat=NS(completions=NS(create=create)))
    assert client.complete(MESSAGES)[0] == 'answer=1'
    assert calls[0]['stream'] is False
    assert ('provider' in calls[0]['extra_body']) == (kind == 'openrouter')
    if kind == 'openrouter':
        assert calls[0]['extra_body']['provider'] == {'sort': 'latency', 'allow_fallbacks': True}


def test_deadline_clamps_watchdog_and_request_timeout(monkeypatch):
    stream = Stream([(2, chunk('answer=42'))])
    client, _, calls = make_client(monkeypatch, [stream])
    stop = DeadlineStop(threading.Event(), time.monotonic() + .04)
    started = time.monotonic()
    with pytest.raises(llm.StallError):
        GuardedClient(client, stop).complete(MESSAGES)
    assert time.monotonic() - started < .3
    assert 0 < calls[0]['timeout'].read <= .04
    assert stream.closed.wait(.3)


def test_deadline_during_headers_closes_late_stream(monkeypatch):
    stream = good()
    def delayed_headers():
        time.sleep(.12)
        return stream
    client, _, _ = make_client(monkeypatch, [delayed_headers])
    with pytest.raises(llm.StallError):
        client.complete(MESSAGES, remaining_time=.03)
    assert stream.closed.wait(.3)


def test_budget_clamps_tokens_and_refuses_unaffordable_input(monkeypatch):
    client, _, calls = make_client(monkeypatch, [good()])
    client.complete(MESSAGES, max_cost=.00002)
    assert calls[0]['max_tokens'] <= 152
    with pytest.raises(llm.NoLLM):
        client.complete(MESSAGES, max_cost=.00000001)
    assert len(calls) == 1


@pytest.mark.parametrize('key,value', [('stream', 1), ('first_token_timeout', 0),
    ('stall_timeout', -1), ('hedge_after', 0), ('max_hedges', True),
    ('max_hedges', -1), ('provider_routing', []),
    ('max_inflight_requests', 0), ('max_inflight_requests', True),
    ('stream_max_seconds', 0), ('stream_max_seconds', -1),
    ('stream_max_seconds', True), ('stream_max_seconds', float('inf'))])
def test_stall_config_validation(key, value):
    config = load_config()
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.mark.parametrize('code', [503, 'server_error', 'provider_error'])
def test_sdk_stream_errors_retry_using_body(monkeypatch, code):
    import httpx
    from openai import APIError
    error = APIError('upstream failed', request=httpx.Request('POST', 'https://offline.test'),
                     body={'code': code, 'message': 'upstream failed'})
    failed = Stream([(0, error)])
    client, config, calls = make_client(monkeypatch, [failed, good()])
    monkeypatch.setattr('solver.strategies.run_program', lambda *a, **k: {'answer': 42})
    answer, _, _, _ = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer == '42' and len(calls) == 2
    assert failed.closed.wait(.3)


@pytest.mark.parametrize('usage', [None, NS(cost=0), NS(prompt_tokens=1, completion_tokens=0)])
def test_stalled_reservation_prevents_retry_over_budget(monkeypatch, usage):
    stalled = Stream([(0, chunk('billable', usage=usage)), (2, chunk('answer=42'))])
    client, config, calls = make_client(monkeypatch, [stalled, good()])
    answer, cost, attempts, _ = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.00002, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer is None and len(calls) == 1
    assert .000019 <= cost <= .00002
    assert sum(a['cost'] for a in attempts) == pytest.approx(cost)


def test_usage_chunk_with_choice_preserves_finish_and_reasoning(monkeypatch):
    final = chunk(usage=NS(cost=.004, prompt_tokens=10, completion_tokens=20,
                          completion_tokens_details=NS(reasoning_tokens=15)))
    final.choices = [NS(delta=NS(content='', reasoning=None), finish_reason='length')]
    client, _, _ = make_client(monkeypatch, [Stream([(0, chunk('answer=42')), (0, final)])])
    text, cost, metadata = client.complete(MESSAGES)
    assert text == 'answer=42' and cost == .004
    assert metadata['finish_reason'] == 'length' and metadata['reasoning_tokens'] == 15


def test_hedge_failure_does_not_discard_valid_primary(monkeypatch):
    primary = Stream([(0, chunk(reasoning='working')), (.06, chunk('primary')),
                      (0, chunk(usage=NS(cost=.001)))])
    failure = RuntimeError('provider error')
    failure.status_code = 503
    client, _, calls = make_client(monkeypatch, [primary, failure])
    client.hedge_after = .02
    text, cost, metadata = client.complete(MESSAGES, max_cost=.03)
    assert text == 'primary' and len(calls) == 2
    assert cost > .001 and metadata['hedges'] == 1


def test_primary_completion_cancels_duplicate_during_headers(monkeypatch):
    primary = Stream([(0, chunk(reasoning='working')), (.05, chunk('primary')),
                      (0, chunk(usage=NS(cost=.001)))])
    duplicate = good('duplicate', .009)
    def headers():
        time.sleep(.1)
        return duplicate
    client, _, _ = make_client(monkeypatch, [primary, headers])
    client.hedge_after = .02
    text, cost, metadata = client.complete(MESSAGES)
    assert text == 'primary' and metadata['hedges'] == 1
    assert duplicate.closed.wait(.3)
    assert client.total_cost == pytest.approx(cost)
    assert cost == pytest.approx(.00152012)


def test_partial_usage_does_not_release_active_request_reservation(monkeypatch):
    primary = Stream([(0, chunk(reasoning='working', usage=NS(cost=.000001))),
                      (.05, chunk('primary')), (0, chunk(usage=NS(cost=.000001)))])
    client, _, calls = make_client(monkeypatch, [primary, good('duplicate')])
    client.hedge_after = .02
    # Input=.00000012; 99 output tokens reserve .00001287. There is no
    # remaining allowance for another minimum 64-token request.
    assert client.complete(MESSAGES, max_cost=.000013)[0] == 'primary'
    assert len(calls) == 1


def test_zero_max_hedges_disables_duplicates(monkeypatch):
    primary = Stream([(0, chunk(reasoning='working')), (.05, chunk('primary'))])
    client, _, calls = make_client(monkeypatch, [primary])
    client.hedge_after, client.max_hedges = .02, 0
    assert client.complete(MESSAGES)[0] == 'primary'
    assert len(calls) == 1


def test_invalid_duplicate_does_not_win(monkeypatch):
    primary = Stream([(0, chunk(reasoning='working')), (.05, chunk('primary')),
                      (0, chunk(usage=NS(cost=.001)))])
    client, _, calls = make_client(monkeypatch, [primary, good('   ', .002)])
    client.hedge_after = .02
    text, cost, _ = client.complete(MESSAGES)
    assert text == 'primary' and cost == pytest.approx(.003)
    assert len(calls) == 2


def test_pipeline_clamps_request_to_global_deadline(monkeypatch, tmp_path, data_dir):
    from solver.pipeline import Pipeline
    monkeypatch.chdir(tmp_path)
    stalled = Stream([(2, chunk('answer=42'))])
    client, config, calls = make_client(monkeypatch, [stalled])
    config.update(strategy='program', question_timeout=10, total_time_limit=.15,
                  time_guard_margin=.01, max_attempts=1)
    pipeline = Pipeline(client, config=config)
    row = pipeline.solve({'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'integer'}, data_dir)
    assert row['answer'] == '0'
    assert 0 < calls[0]['timeout'].read <= .14
    assert stalled.closed.wait(.3)


def test_successful_request_cost_retained_when_execution_raises(monkeypatch):
    client, config, _ = make_client(monkeypatch, [good()])
    def broken_execution(*args, **kwargs):
        raise ValueError('execution failed')
    monkeypatch.setattr('solver.strategies.run_program', broken_execution)
    answer, cost, attempts, error = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.03, stopped=threading.Event(), on_event=lambda *a: None)
    assert answer is None and error is not None
    assert cost == pytest.approx(.001)
    assert attempts[0]['cost'] == pytest.approx(.001)
    assert attempts[0]['usage']['prompt_tokens'] == 10


@pytest.mark.parametrize('status', [402, 503])
def test_sdk_http_status_not_lost_to_symbolic_error_body(monkeypatch, status):
    import httpx
    from openai import APIStatusError
    request = httpx.Request('POST', 'https://offline.test')
    response = httpx.Response(status, request=request)
    error = APIStatusError('provider failed', response=response, body={'code': 'unknown'})
    client, config, calls = make_client(monkeypatch, [error, good()])
    monkeypatch.setattr('solver.strategies.run_program', lambda *a, **k: {'answer': 42})
    stop = threading.Event()
    answer, _, _, _ = program(client=client, messages=MESSAGES,
        query={'id': 'q', 'query': 'integer'}, path='unused', config=config,
        budget=.03, stopped=stop, on_event=lambda *a: None)
    if status == 402:
        assert answer is None and stop.is_set() and client.credit_failed.is_set()
        assert len(calls) == 1
    else:
        assert answer == '42' and len(calls) == 2


def test_provider_error_preserved_while_stream_cleanup_runs(monkeypatch):
    error = RuntimeError('bad request')
    error.status_code = 400
    class SlowClose(Stream):
        def close(self):
            time.sleep(.16)
            super().close()
    client, _, _ = make_client(monkeypatch, [SlowClose([(0, error)])])
    client.hedge_after = .01
    with pytest.raises(RuntimeError, match='bad request') as raised:
        client.complete(MESSAGES, max_cost=.000009, remaining_time=.02)
    assert raised.value.status_code == 400


def test_expired_scheduled_worker_never_sends_request(monkeypatch):
    client, _, calls = make_client(monkeypatch, [good()])
    original_start = threading.Thread.start
    def delayed_start(thread):
        time.sleep(.05)
        original_start(thread)
    monkeypatch.setattr(threading.Thread, 'start', delayed_start)
    with pytest.raises(llm.StallError) as raised:
        client.complete(MESSAGES, remaining_time=.02)
    time.sleep(.05)
    assert calls == []
    assert raised.value.cost == 0


def test_openrouter_sdk_has_no_unbudgeted_internal_retries(monkeypatch):
    import httpx
    from openai import APIStatusError
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline')
    config = load_config()
    config.update(api_max_retries=1, hedge_after=None, first_token_timeout=2)
    client = llm.LLMClient(config)
    calls = []
    def transport(request):
        calls.append(request)
        return httpx.Response(503, json={'error': {'message': 'unavailable'}}, request=request)
    http = httpx.Client(transport=httpx.MockTransport(transport))
    client.client = client.client.with_options(http_client=http)
    try:
        with pytest.raises(APIStatusError):
            client.complete(MESSAGES)
        assert len(calls) == 1
    finally:
        http.close()


@pytest.mark.parametrize('usage', [NS(cost=0), NS(prompt_tokens=1, completion_tokens=0)])
def test_cancelled_hedge_retains_partial_usage_reservation(monkeypatch, usage):
    loser = Stream([(0, chunk('billable', usage=usage)), (2, chunk('tail'))])
    client, _, calls = make_client(monkeypatch, [loser, good()], hedge_after=.02)
    text, cost, metadata = client.complete(MESSAGES)
    assert text == 'answer=42' and len(calls) == 2
    assert metadata['request_costs'][0] == pytest.approx(.00052012)
    assert cost == pytest.approx(client.total_cost) == pytest.approx(.00152012)


def test_global_request_worker_cap_survives_header_timeouts(monkeypatch):
    release = threading.Event()
    streams = [good(), good()]
    def headers(stream):
        def blocked():
            release.wait()
            return stream
        return blocked
    first, _, first_calls = make_client(monkeypatch, [headers(streams[0])], max_inflight_requests=2)
    second, _, second_calls = make_client(monkeypatch, [headers(streams[1]), good()], max_inflight_requests=2)
    try:
        for client in (first, second, first, second, first):
            started = time.monotonic()
            with pytest.raises(llm.StallError):
                client.complete(MESSAGES, remaining_time=.025)
            assert time.monotonic() - started < .15
        assert len(first_calls) == len(second_calls) == 1
    finally:
        release.set()
    assert all(stream.closed.wait(.5) for stream in streams)
    assert second.complete(MESSAGES, remaining_time=.2)[0] == 'answer=42'


@pytest.mark.parametrize('mode', ['timeout', 'success'])
def test_blocking_close_does_not_hold_completion_past_deadline(monkeypatch, mode):
    release = threading.Event()
    closing = threading.Event()
    class BlockingClose(Stream):
        def close(self):
            closing.set()
            release.wait()
            super().close()
    steps = good().steps if mode == 'success' else [(2, chunk('late'))]
    stream = BlockingClose(steps)
    client, _, _ = make_client(monkeypatch, [stream])
    started = time.monotonic()
    try:
        if mode == 'success':
            assert client.complete(MESSAGES, remaining_time=.03)[0] == 'answer=42'
        else:
            with pytest.raises(llm.StallError):
                client.complete(MESSAGES, remaining_time=.03)
        assert time.monotonic() - started < .15
        assert closing.wait(.2)
        assert not stream.closed.is_set()
    finally:
        release.set()
    assert stream.closed.wait(.5)


@pytest.mark.parametrize('tail', ['stall', 'error'])
def test_finished_answer_survives_missing_terminal_usage(monkeypatch, tail):
    last = (2, chunk(usage=NS(cost=.001))) if tail == 'stall' else (0, RuntimeError('tail failed'))
    stream = Stream([(0, chunk('answer=42', finish='stop', usage=NS(cost=0))), last])
    client, _, _ = make_client(monkeypatch, [stream])
    text, cost, metadata = client.complete(MESSAGES, remaining_time=.03)
    assert text == 'answer=42' and metadata['finish_reason'] == 'stop'
    assert cost == pytest.approx(.00052012)
    assert client.total_cost == pytest.approx(cost)


def test_completed_primary_survives_simultaneous_hedge_credit_failure(monkeypatch):
    primary_done = threading.Event()
    duplicate_done = threading.Event()
    duplicate_started = threading.Event()
    class Primary(Stream):
        def __iter__(self):
            yield chunk(reasoning='working')
            assert duplicate_started.wait(.5)
            yield chunk('answer=42', finish='stop')
            yield chunk(usage=NS(cost=.001))
    error = RuntimeError('insufficient credit')
    error.status_code = 402
    def duplicate():
        duplicate_started.set()
        assert primary_done.wait(.5)
        raise error
    client, _, calls = make_client(monkeypatch, [Primary([]), duplicate], hedge_after=.01)
    original_request = client._request
    def request(state, condition, kwargs):
        original_request(state, condition, kwargs)
        if state.content == 'answer=42':
            primary_done.set()
        else:
            duplicate_done.set()
    client._request = request
    # Let both workers publish before the coordinator resumes from its wait.
    original_condition = threading.Condition
    class SlowCoordinator(original_condition):
        def wait(self, timeout=None):
            result = super().wait(timeout)
            if duplicate_started.is_set():
                self.release()
                try:
                    assert duplicate_done.wait(.5)
                finally:
                    self.acquire()
            return result
    # Replace only the coordinator's condition, leaving Thread/Event internals alone.
    coordinator_created = False
    def condition(*args, **kwargs):
        nonlocal coordinator_created
        if not coordinator_created:
            coordinator_created = True
            return SlowCoordinator(*args, **kwargs)
        return original_condition(*args, **kwargs)
    monkeypatch.setattr(llm.threading, 'Condition', condition)
    text, cost, metadata = client.complete(MESSAGES)
    assert text == 'answer=42' and len(calls) == 2
    assert client.credit_failed.is_set()
    assert metadata['request_costs'] == pytest.approx([.001, .00052012])
    assert cost == pytest.approx(client.total_cost)
    with pytest.raises(llm.InsufficientCredit):
        client.complete(MESSAGES)
    assert len(calls) == 2


def test_cancelled_zero_usage_cannot_fund_another_hedge(monkeypatch):
    primary = Stream([(0, chunk('partial', usage=NS(cost=0))), (2, chunk('late'))])
    duplicate = Stream([(.025, chunk(reasoning='working')),
                        (.025, chunk(reasoning='working')),
                        (.025, chunk('answer=42', finish='stop')),
                        (0, chunk(usage=NS(cost=.000001)))])
    client, _, calls = make_client(monkeypatch, [primary, duplicate, good()],
                                  hedge_after=.02, max_hedges=2, stall_timeout=.04,
                                  first_token_timeout=.04)
    text, cost, metadata = client.complete(MESSAGES, max_cost=.00054)
    assert text == 'answer=42' and len(calls) == 2
    assert metadata['request_costs'][0] == pytest.approx(.00052012)
    assert cost == pytest.approx(.00052112)
