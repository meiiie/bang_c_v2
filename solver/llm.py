"""Configurable compatible API client and explicitly offline program provider."""
import copy
import errno
import socket
import ssl
import httpx
import json
import os
from pathlib import Path
import re
import queue
import threading
import time
from dataclasses import dataclass
from openai import APIError, APIConnectionError


class _RequestSlots:
    """Process-wide slots remain occupied even after watchdog cancellation."""
    def __init__(self):
        self.condition = threading.Condition()
        self.active = 0

    def acquire(self, limit, deadline, wait=True):
        with self.condition:
            while self.active >= limit:
                remaining = deadline - time.monotonic()
                if not wait or remaining <= 0:
                    return False
                self.condition.wait(None if remaining == float('inf') else remaining)
            if time.monotonic() >= deadline:
                return False
            self.active += 1
            return True

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()


_request_slots = _RequestSlots()
_cleanup_queue = queue.SimpleQueue()
_cleanup_lock = threading.Lock()
_cleanup_started = False


def _cleanup_worker():
    while True:
        response = _cleanup_queue.get()
        try:
            response.close()
        except Exception:
            pass  # Cleanup must not hide the original provider failure.


class StallError(TimeoutError):
    """Transient first-token, stream-idle or request-deadline failure."""


class ProviderStreamError(RuntimeError):
    def __init__(self, error):
        super().__init__(str(error))
        code = error.get('code') if isinstance(error, dict) else getattr(error, 'code', None)
        code = {'server_error': 502, 'provider_error': 502,
                'rate_limit_exceeded': 429, 'insufficient_credits': 402}.get(code, code)
        try:
            self.status_code = int(code)
        except (TypeError, ValueError):
            self.status_code = None


def _ignored_providers_error(exc):
    text = str(exc).lower()
    return (getattr(exc, 'status_code', None) in (400, 404)
            and 'ignored' in text and 'provider' in text)


def is_transient(exc):
    """Transport failures and retryable provider statuses, never local disk errors."""
    if _ignored_providers_error(exc):
        return True
    if isinstance(exc, (StallError, APIConnectionError, httpx.TransportError,
                        ConnectionError, TimeoutError, socket.gaierror, ssl.SSLError)):
        return True
    if isinstance(exc, OSError) and exc.errno in {
            errno.ENETDOWN, errno.ENETUNREACH, errno.ENETRESET, errno.ECONNABORTED,
            errno.ECONNRESET, errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ETIMEDOUT,
            errno.EPIPE, errno.EHOSTDOWN}:
        return True
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    name, text = type(exc).__name__.lower(), str(exc).lower()
    return any(k in name or k in text for k in
               ('timeout', 'timed out', 'connection', 'rate limit', '429', '502', '503', '504', '500'))


_provider_failures = {}
_provider_lock = threading.Lock()


def _failed_providers():
    with _provider_lock:
        now = time.monotonic()
        for provider, expiry in list(_provider_failures.items()):
            if expiry <= now:
                del _provider_failures[provider]
        return list(_provider_failures)


@dataclass
class _Request:
    started: float
    max_tokens: int
    deadline: float
    client: object = None
    provider: str = None
    response: object = None
    usage: object = None
    content: str = ''
    finish_reason: str = None
    error: Exception = None
    first_token: bool = False
    sent: bool = False
    cancelled: bool = False
    done: bool = False
    completed_at: float = 0
    content_finished: bool = False
    usage_final: bool = False
    failure_recorded: bool = False

    def __post_init__(self):
        self.last_chunk = self.started


class NoLLM(RuntimeError):
    pass


class InsufficientCredit(NoLLM):
    pass


def is_credit_error(exc):
    text = str(exc).lower()
    return isinstance(exc, InsufficientCredit) or getattr(exc, 'status_code', None) == 402 or 'insufficient credit' in text or 'insufficient_credit' in text or 'insufficient funds' in text


class LLMClient:
    supports_deadlines = True

    def __init__(self, config=None):
        from .config import load_config, validate_config
        config = load_config() if config is None else validate_config(config)
        from openai import OpenAI
        self.model = config['model']
        key = os.environ.get(config['api_key_env'])
        if not key:
            raise NoLLM(f"Missing API key environment variable: {config['api_key_env']}")
        self.provider_kind = config['provider_kind']
        # OpenRouter retries must pass through strategies so every attempt is
        # deadline checked and budget reserved; SDK retries cannot be cancelled.
        transport_timeout = config['request_timeout']
        if config['stream']:
            # HTTPX timeouts bound each operation/read, not total stream time.
            # Allow headers/first token their full window; the watchdog switches
            # to the stricter idle window after the first token.
            transport_timeout = httpx.Timeout(transport_timeout, read=max(
                config['first_token_timeout'], config['stall_timeout']))
        self._client_settings = dict(api_key=key, base_url=config['base_url'],
                                     timeout=transport_timeout,
                                     max_retries=0 if self.provider_kind == 'openrouter' else int(config.get('api_max_retries', 2)))
        self._client_factory = OpenAI
        self.client = self._client_factory(**self._client_settings)
        self._pool_lock = threading.Lock()
        self._pool_users = {}
        self._retired_clients = {}
        self._routing_retry = threading.local()
        self.fresh_client_on_transient = config['fresh_client_on_transient']
        self.provider_ignore = config['provider_ignore']
        self.provider_failure_ttl = config['provider_failure_ttl']
        self.rotate_provider_sort = config['rotate_provider_sort']
        self.input_price = config['input_price']
        self.output_price = config['output_price']
        self.reasoning = config['reasoning']
        self.max_tokens = config['max_tokens']
        self.stream = config['stream']
        self.first_token_timeout = config['first_token_timeout']
        self.stall_timeout = config['stall_timeout']
        self.hedge_after = config['hedge_after']
        self.max_hedges = config['max_hedges']
        self.provider_routing = config['provider_routing']
        self.request_timeout = config['request_timeout']
        self.stream_max_seconds = config['stream_max_seconds']
        self.max_inflight_requests = config['max_inflight_requests']
        self.credit_failed = threading.Event()
        self.total_cost = 0.0
        self.lock = threading.Lock()

    def extra_body(self, reasoning=None):
        reasoning = self.reasoning if reasoning is None else reasoning
        if self.provider_kind == 'openrouter':
            body = {'usage': {'include': True}, 'reasoning':
                    {'enabled': False} if reasoning == 'off' else {'effort': reasoning}}
            if self.provider_routing is not None:
                body['provider'] = copy.deepcopy(self.provider_routing)
            routing = body.get('provider', {})
            if not self.provider_ignore:
                routing.pop('ignore', None)
            ignored = _failed_providers() if self.provider_ignore else []
            if self.provider_ignore and routing.get('ignore'):
                routing['ignore'] = routing['ignore'][:1]
            retries = getattr(self._routing_retry, 'count', 0)
            if ignored or (retries and self.rotate_provider_sort):
                routing = body.setdefault('provider', {'allow_fallbacks': True})
                if ignored:
                    routing['ignore'] = list(dict.fromkeys([*routing.get('ignore', []), *ignored]))[:1]
                if retries and self.rotate_provider_sort:
                    initial = (self.provider_routing or {}).get('sort', 'latency')
                    order = ('throughput', 'latency') if initial == 'throughput' else ('latency', 'throughput')
                    routing['sort'] = order[retries % 2]
            return body
        if self.provider_kind == 'zai':
            return {'thinking': {'type': 'disabled' if reasoning == 'off' else 'enabled'}}
        return {}

    def _acquire_client(self):
        with self._pool_lock:
            client = self.client
            identity = id(client)
            self._pool_users[identity] = self._pool_users.get(identity, 0) + 1
            return client

    def _release_client(self, client):
        with self._pool_lock:
            identity = id(client)
            users = self._pool_users[identity] - 1
            if users:
                self._pool_users[identity] = users
            else:
                del self._pool_users[identity]
                self._close(self._retired_clients.pop(identity, None))

    def _failed_request(self, state):
        if state.failure_recorded or not state.error or not is_transient(state.error):
            return
        state.failure_recorded = True
        if (self.provider_ignore and not _ignored_providers_error(state.error)
                and self.provider_kind == 'openrouter' and isinstance(state.provider, str) and state.provider):
            with _provider_lock:
                _provider_failures[state.provider] = time.monotonic() + self.provider_failure_ttl
        if self.fresh_client_on_transient:
            with self._pool_lock:
                if self.client is state.client:
                    # Existing workers retain leases; closing waits for their exit.
                    self.client = self._client_factory(**self._client_settings)
                    if self._pool_users.get(id(state.client), 0):
                        self._retired_clients[id(state.client)] = state.client
                    else:
                        self._close(state.client)

    def _cost(self, usage, input_tokens, max_tokens):
        cost = getattr(usage, 'cost', None)
        if cost is not None:
            return float(cost)
        prompt = getattr(usage, 'prompt_tokens', None)
        output = getattr(usage, 'completion_tokens', None)
        return ((input_tokens if prompt is None else prompt)*self.input_price +
                (max_tokens if output is None else output)*self.output_price)/1e6

    def _tokens(self, input_tokens, allowance):
        max_tokens = self.max_tokens
        if allowance is not None:
            available = allowance - input_tokens*self.input_price/1e6
            if available < 0:
                raise NoLLM('Insufficient remaining question budget')
            if self.output_price:
                max_tokens = min(max_tokens, int(available*1e6/self.output_price))
            if max_tokens < 64:
                raise NoLLM('Insufficient remaining question budget')
        return max_tokens

    @staticmethod
    def _close(response):
        global _cleanup_started
        if response is not None and hasattr(response, 'close'):
            with _cleanup_lock:
                if not _cleanup_started:
                    for _ in range(2):
                        threading.Thread(target=_cleanup_worker, daemon=True,
                                         name='llm-cleanup').start()
                    _cleanup_started = True
            _cleanup_queue.put(response)

    def _request_cost(self, state, input_tokens):
        if not state.sent:
            return 0.0
        cost = self._cost(state.usage, input_tokens, state.max_tokens)
        if state.usage_final and state.done and not state.cancelled and state.error is None:
            return cost
        return max(cost, self._cost(None, input_tokens, state.max_tokens))

    def _timeout_error(self, state, now):
        if now >= state.deadline:
            return StallError('Question/run or request deadline reached')
        if self.stream:
            if not state.first_token and now-state.started >= self.first_token_timeout:
                return StallError('Timed out waiting for first token')
            if state.first_token and now-state.last_chunk >= self.stall_timeout:
                return StallError('Stream stalled between chunks')
        return None

    def _request(self, state, condition, kwargs):
        """Retry an exhausted ignore filter once within the original deadline."""
        try:
            for attempt in range(2):
                try:
                    self._request_once(state, condition, kwargs)
                    break
                except Exception as exc:
                    # The SDK raises APIError before yielding an SSE error frame.
                    if (self.stream and isinstance(exc, APIError) and getattr(exc, 'status_code', None) is None
                            and isinstance(exc.body, dict) and exc.body.get('code') is not None):
                        exc = ProviderStreamError(exc.body)
                    if is_credit_error(exc):
                        self.credit_failed.set()
                        exc = InsufficientCredit(f'Provider insufficient credit (402): {exc}')
                    if (attempt == 0 and self.provider_kind == 'openrouter'
                            and self.provider_ignore and _ignored_providers_error(exc)):
                        with _provider_lock:
                            _provider_failures.clear()
                        kwargs['extra_body'].get('provider', {}).pop('ignore', None)
                        with condition:
                            owned_response, state.response = state.response, None
                            state.provider = None
                            state.usage = None
                            state.content = ''
                            state.finish_reason = None
                            state.first_token = False
                            state.content_finished = False
                            state.usage_final = False
                        self._close(owned_response)
                        continue
                    with condition:
                        if not state.cancelled:
                            state.error = exc
                    break
        finally:
            with condition:
                owned_response, state.response = state.response, None
                state.done = True
                state.completed_at = time.monotonic()
                condition.notify_all()
            self._close(owned_response)
            self._release_client(state.client)
            _request_slots.release()

    def _request_once(self, state, condition, kwargs):
        """Only this daemon reads its stream; the coordinator may cancel it."""
        with condition:
            if state.cancelled:
                return
            timeout_error = self._timeout_error(state, time.monotonic())
            if timeout_error:
                raise timeout_error
            # Thread scheduling may have consumed part of the original allowance.
            remaining = state.deadline-time.monotonic()
            if remaining <= 0:
                raise StallError('Question/run deadline reached before request')
            if self.stream:
                kwargs['timeout'] = httpx.Timeout(
                    min(self.request_timeout, remaining),
                    read=min(max(self.first_token_timeout, self.stall_timeout), remaining))
            else:
                kwargs['timeout'] = min(kwargs['timeout'], remaining)
            state.sent = True
        response = state.client.chat.completions.create(**kwargs)
        with condition:
            cancelled = state.cancelled
            if not cancelled:
                state.response = response
                state.provider = getattr(response, 'provider', None) or state.provider
                timeout_error = self._timeout_error(state, time.monotonic())
            else:
                timeout_error = None
        if cancelled:
            self._close(response)
            return
        if timeout_error:
            raise timeout_error
        # A completion object also supports compatible transports that don't stream.
        if not self.stream or hasattr(response, 'choices'):
            choice = response.choices[0]
            with condition:
                state.usage = response.usage
                state.content = choice.message.content or ''
                state.finish_reason = getattr(choice, 'finish_reason', None)
                state.usage_final = state.usage is not None
        else:
            for chunk in response:
                with condition:
                    if state.cancelled:
                        return
                    state.provider = getattr(chunk, 'provider', None) or state.provider
                    now = time.monotonic()
                    timeout_error = self._timeout_error(state, now)
                    if timeout_error:
                        raise timeout_error
                    state.last_chunk = now
                    usage = getattr(chunk, 'usage', None)
                    if usage is not None:
                        state.usage = usage
                        state.usage_final = True
                    error = getattr(chunk, 'error', None)
                    if error:
                        raise ProviderStreamError(error)
                    for choice in chunk.choices:
                        delta = choice.delta
                        content = getattr(delta, 'content', None)
                        # OpenRouter exposes both reasoning and structured reasoning details.
                        token = content or any(getattr(delta, key, None) for key in
                                               ('reasoning', 'reasoning_content', 'reasoning_details'))
                        if token:
                            state.first_token = True
                            if usage is None:
                                state.usage_final = False
                        if content:
                            state.content += content
                        if choice.finish_reason is not None:
                            state.finish_reason = choice.finish_reason
                            state.content_finished = True
                    condition.notify_all()
            with condition:
                state.usage_final = state.usage_final and not state.cancelled

    def complete(self, messages, max_cost=None, temperature=0, reasoning=None, remaining_time=None):
        if self.credit_failed.is_set():
            raise InsufficientCredit('Provider insufficient credit; run stopped')
        # Bytes conservatively bound input tokens, including non-ASCII workbook labels.
        input_tokens = sum(len(m['content'].encode('utf-8')) for m in messages)
        max_tokens = self._tokens(input_tokens, max_cost)
        started = time.monotonic()
        # A progressing stream may outlive request_timeout. Only the caller's
        # question/run deadline and optional stream cap bound its total time.
        duration = (self.stream_max_seconds if self.stream_max_seconds is not None
                    else float('inf')) if self.stream else self.request_timeout
        if remaining_time is not None:
            duration = min(duration, remaining_time)
        if duration <= 0:
            raise StallError('Question/run deadline reached')
        deadline = started + duration
        condition = threading.Condition()
        states = []

        def launch(tokens, wait=True):
            # Do not hold the coordinator lock while waiting for other calls.
            condition.release()
            try:
                acquired = _request_slots.acquire(self.max_inflight_requests, deadline, wait)
            finally:
                condition.acquire()
            if not acquired:
                return False
            if self.credit_failed.is_set():
                _request_slots.release()
                return False
            now = time.monotonic()
            state = _Request(now, tokens, deadline, client=self._acquire_client())
            states.append(state)
            kwargs = dict(model=self.model, messages=messages, max_tokens=tokens,
                          temperature=temperature, extra_body=self.extra_body(reasoning),
                          stream=self.stream, timeout=deadline-now)
            if self.stream:
                kwargs['stream_options'] = {'include_usage': True}
            try:
                threading.Thread(target=self._request, args=(state, condition, kwargs),
                                 daemon=True, name='llm-request').start()
            except BaseException:
                self._release_client(state.client)
                _request_slots.release()
                raise
            return True

        winner, failure = None, None
        with condition:
            if not launch(max_tokens):
                error = (InsufficientCredit('Provider insufficient credit; run stopped')
                         if self.credit_failed.is_set() else
                         StallError('Request worker capacity deadline reached'))
                error.cost = 0.0
                error.usage = {'request_costs': [], 'hedges': 0}
                raise error
            next_hedge = started + self.hedge_after if self.hedge_after is not None else float('inf')
            while True:
                now = time.monotonic()
                valid = [s for s in states if s.content.strip() and
                         ((s.done and s.error is None) or
                          (s.content_finished and (s.done or s.cancelled)))]
                if valid:
                    winner = min(valid, key=lambda s: s.completed_at)
                    break
                if self.credit_failed.is_set():
                    failure = InsufficientCredit('Provider insufficient credit; run stopped')
                    break
                for state in states:
                    if state.done or state.cancelled:
                        self._failed_request(state)
                        continue
                    if state.error is None:
                        state.error = self._timeout_error(state, now)
                    if state.error is not None:
                        self._failed_request(state)
                        state.cancelled = True
                finished = [s for s in states if s.cancelled and s.content_finished and s.content.strip()]
                if finished:
                    winner = finished[0]
                    break
                active = [s for s in states if not s.done and not s.cancelled]
                if not active:
                    # Preserve empty/length responses for the strategy's reasoning fallback.
                    empty = [s for s in states if s.done and s.error is None]
                    if empty:
                        winner = min(empty, key=lambda s: s.completed_at)
                    else:
                        failure = states[-1].error or StallError('Request cancelled')
                    break
                if now >= next_hedge and len(states)-1 < self.max_hedges:
                    # Reserve every launched request's estimate until authoritative usage arrives.
                    reserved = sum(self._request_cost(s, input_tokens) if s.sent else
                                   self._cost(None, input_tokens, s.max_tokens) for s in states)
                    allowance = None if max_cost is None else max_cost-reserved
                    # Give a duplicate a full first-token window; near deadlines, avoid extra spend.
                    if deadline-now >= self.first_token_timeout:
                        try:
                            tokens = self._tokens(input_tokens, allowance)
                        except NoLLM:
                            pass
                        else:
                            launch(tokens, wait=False)
                    next_hedge = now + self.hedge_after
                # Queue cleanup without holding up the surviving request.
                for state in states:
                    if state.cancelled and state.response is not None:
                        self._close(state.response)
                        state.response = None
                wake = deadline
                for state in active:
                    if self.stream:
                        if not state.first_token:
                            wake = min(wake, state.started+self.first_token_timeout)
                        else:
                            wake = min(wake, state.last_chunk+self.stall_timeout)
                if len(states)-1 < self.max_hedges:
                    wake = min(wake, next_hedge)
                condition.wait(max(.001, wake-time.monotonic()))
            # Freeze accounting before cancellation. Late usage cannot charge another question.
            for state in states:
                self._failed_request(state)
            costs = [self._request_cost(s, input_tokens) for s in states]
            expense = sum(costs)
            responses = []
            for state in states:
                state.cancelled = True
                if state.response is not None:
                    responses.append(state.response)
                    state.response = None
            usage = winner.usage if winner else None
            metadata = self._metadata(usage, winner.finish_reason if winner else None)
            metadata['request_costs'] = costs
            metadata['hedges'] = len(states)-1
        for response in responses:
            self._close(response)
        with self.lock:
            self.total_cost += expense
        if failure is not None:
            if is_transient(failure):
                self._routing_retry.count = getattr(self._routing_retry, 'count', 0) + 1
            failure.cost = expense
            failure.usage = metadata
            raise failure
        self._routing_retry.count = 0
        return winner.content, expense, metadata

    @staticmethod
    def _metadata(usage, finish_reason):
        details = getattr(usage, 'completion_tokens_details', None)
        return {'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                'completion_tokens': getattr(usage, 'completion_tokens', None),
                'reasoning_tokens': getattr(details, 'reasoning_tokens', None),
                'finish_reason': finish_reason}


class MockClient:
    def __init__(self, programs_file=None):
        self.path = Path(programs_file) if programs_file else None
        self.programs = json.loads(self.path.read_text()) if self.path and self.path.is_file() else None

    def complete(self, messages, max_cost=None, temperature=0, reasoning=None):
        if self.path is None: raise NoLLM('No mock programs supplied')
        text = '\n'.join(m['content'] for m in messages)
        match = re.search(r'Query ID: ([\w-]+)', text)
        if not match: raise NoLLM('Query ID missing')
        query_id = match.group(1)
        if self.programs is not None:
            if query_id not in self.programs: raise NoLLM(f'No program for {query_id}')
            return self.programs[query_id], 0.0
        path = self.path / f'{query_id}.py'
        if not path.exists(): raise NoLLM(f'No program for {query_id}')
        return path.read_text(), 0.0
