"""Versioned configuration: defaults < file/profile < environment < CLI."""
import copy
import csv
from concurrent.futures import Future
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time

DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'configs/default.json'


def _merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def _value(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config, overrides):
    result = copy.deepcopy(config)
    for item in overrides:
        key, sep, raw = item.partition('=')
        if not sep:
            raise ValueError('--set requires key=value')
        target = result
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        if parts[-1] not in target:
            raise ValueError(f'Unknown configuration key: {key}')
        target[parts[-1]] = _value(raw)
    if any(item.startswith('retries=') for item in overrides) and not any(item.startswith('max_attempts=') for item in overrides):
        result['max_attempts'] = result['retries'] + 1
    return validate_config(result)


def validate_config(config):
    if config['schema_version'] != 1:
        raise ValueError('Unsupported config schema_version')
    if (config['profile'] == 'contest' or config['provider_kind'] == 'openrouter') and config['model'] != 'qwen/qwen3.7-flash':
        raise ValueError('Contest permits only qwen/qwen3.7-flash')
    if config['provider_kind'] not in ('openrouter', 'zai', 'openai_compatible'):
        raise ValueError('Unknown provider_kind')
    if config['profile'] == 'contest' and config['provider_kind'] != 'openrouter':
        raise ValueError('Contest requires OpenRouter')
    if not isinstance(config['base_url'], str) or not config['base_url'].startswith(('https://', 'http://')):
        raise ValueError('base_url must be an HTTP(S) URL')
    if not isinstance(config['api_key_env'], str) or not config['api_key_env']:
        raise ValueError('api_key_env must name an environment variable')
    if not 0 < config['budget'] <= .10:
        raise ValueError('Budget must be in (0, 0.10] USD/question')
    if config['reasoning'] not in ('off', 'low', 'medium'):
        raise ValueError('SOLVER_REASONING must be off, low, or medium')
    if config.get('length_fallback_reasoning', 'off') not in (None, 'off', 'low', 'medium'):
        raise ValueError('length_fallback_reasoning must be null, off, low, or medium')
    if config['max_tokens'] < 64:
        raise ValueError('SOLVER_MAX_TOKENS must be at least 64')
    for key in ('votes', 'max_votes', 'workers', 'max_attempts', 'program_workers'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['max_abandoned_workers']) is not int or config['max_abandoned_workers'] < 0:
        raise ValueError('max_abandoned_workers must be a non-negative integer')
    if type(config['max_inflight_requests']) is not int or config['max_inflight_requests'] < 1:
        raise ValueError('max_inflight_requests must be a positive integer')
    for key, choices in (('vote_mode', ('fixed', 'adaptive')), ('serialization', ('pipe', 'compact'))):
        if config[key] not in choices:
            raise ValueError(f'Invalid {key}')
    for key in ('hybrid_text', 'verifier', 'fast_path', 'stream', 'sanity_retry', 'fresh_client_on_transient', 'provider_ignore', 'rotate_provider_sort', 'rescue_pass'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be boolean')
    if type(config['max_transient_retries']) is not int or config['max_transient_retries'] < 0:
        raise ValueError('max_transient_retries must be a non-negative integer')
    if type(config['sanity_max_retries']) is not int or config['sanity_max_retries'] < 0:
        raise ValueError('sanity_max_retries must be a non-negative integer')
    if type(config['max_hedges']) is not int or config['max_hedges'] < 0:
        raise ValueError('max_hedges must be a non-negative integer')
    routing = config['provider_routing']
    if routing is not None and not isinstance(routing, dict):
        raise ValueError('provider_routing must be an object or null')
    if config['hedge_after'] is not None:
        value = config['hedge_after']
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('hedge_after must be positive or null')
    value = config['stream_max_seconds']
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
        raise ValueError('stream_max_seconds must be positive or null')
    value = config['transient_retry_sleep']
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('transient_retry_sleep must be nonnegative')
    if not isinstance(config['retries'], int) or config['retries'] < 0:
        raise ValueError('retries must be nonnegative')
    for key in ('request_timeout', 'program_timeout', 'prompt_char_budget', 'total_time_limit', 'question_timeout', 'first_token_timeout', 'stall_timeout', 'provider_failure_ttl'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not math.isfinite(config['time_guard_margin']) or config['time_guard_margin'] < 0:
        raise ValueError('time_guard_margin must be nonnegative')
    if config['program_timeout'] > 20:
        raise ValueError('Sandbox program timeout cannot exceed 20 seconds')
    if any(not math.isfinite(config[key]) or config[key] < 0 for key in ('input_price', 'output_price')):
        raise ValueError('Token prices must be nonnegative')
    return config


def load_config(path=None, profile=None):
    config = json.loads(DEFAULT_PATH.read_text())
    overlay = json.loads(Path(path).read_text()) if path else {}
    config = _merge(config, overlay)
    profile = profile or os.environ.get('SOLVER_PROFILE') or config['profile']
    if profile:
        if profile not in config['profiles']:
            raise ValueError(f'Unknown profile: {profile}')
        overlay = _merge(overlay, config['profiles'][profile])
        config = _merge(config, config['profiles'][profile])
    # A custom file may override built-in profile defaults without editing that profile.
    file_overlay = json.loads(Path(path).read_text()) if path else {}
    config = _merge(config, {k: v for k, v in file_overlay.items() if k not in ('profiles', 'profile')})
    config['profile'] = profile
    overrides = [f'{key}={os.environ["SOLVER_" + key.upper()]}' for key in config
                 if 'SOLVER_' + key.upper() in os.environ and key not in ('profiles', 'profile')]
    if 'OPENROUTER_BASE_URL' in os.environ:
        overrides.append('base_url=' + os.environ['OPENROUTER_BASE_URL'])
    if 'retries' in overlay and 'max_attempts' not in overlay:
        config['max_attempts'] = config['retries'] + 1
    return apply_overrides(config, overrides)


# Deadline helpers keep stalled provider threads from holding CLI shutdown open.


def launch(fn, *args):
    future = Future()
    def work():
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
    threading.Thread(target=work, daemon=True).start()
    return future


class DeadlineStop:
    def __init__(self, event, deadline):
        self.event, self.deadline = event, deadline

    def is_set(self):
        return self.event.is_set() or time.monotonic() >= self.deadline

    def set(self):
        self.event.set()

    def remaining(self):
        return max(0, self.deadline - time.monotonic())


class GuardedClient:
    def __init__(self, client, stopped):
        self.client, self.stopped = client, stopped

    def complete(self, *args, **kwargs):
        if self.stopped.is_set():
            raise TimeoutError('Run stopped or question deadline reached')
        if getattr(self.client, 'supports_deadlines', False):
            remaining = self.stopped.remaining()
            kwargs['remaining_time'] = min(kwargs.get('remaining_time', remaining), remaining)
        return self.client.complete(*args, **kwargs)


# Emergency answers and atomic, complete CSV snapshots use only the standard library.


def fallback_answer(query):
    text = str(query.get('query', '')).casefold()
    if re.search(r'yes\s*(?:or|/|-)\s*no', text):
        return 'No'
    two = 'two decimal' in text or '2 decimal' in text or '2-decimal' in text or 'rounded to two' in text
    single = '0.00' if two else '0'
    if 'comma' in text:
        # Numeric multi-value answers: count from "two integers" or a backticked template.
        m = re.search(r'\b(two|three|four|five)\b (?:integers|numbers|values)', text)
        n = {'two': 2, 'three': 3, 'four': 4, 'five': 5}[m.group(1)] if m else 0
        t = re.search(r'answer as `([^`]*)`', text)
        if not n and t and ',' in t.group(1):
            n = t.group(1).count(',') + 1
        if n:
            return ', '.join([single] * n)
    if two:
        return '0.00'
    # A generic numeric default also preserves legacy failure behavior.
    return '0'


def normalize_query_ids(queries):
    return [dict(q, id='' if q['id'] is None else str(q['id'])) for q in queries]


def initial_rows(queries, mode='llm'):
    # Valid contest input has unique IDs; duplicate entries collapse deterministically.
    return list({str(q['id']): {'id': str(q['id']), 'answer': '' if mode == 'dump' else fallback_answer(q)}
                 for q in queries}.values())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o755)
    except OSError:
        pass
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'answer'])
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
