"""Bounded code/execute/observe strategy and independent voting wrapper."""
import time
from .config import launch
from .program import run_program
from .formatter import format_answer
from .llm import is_credit_error, is_transient, StallError
from .sanity import check_answer, table_labels, retry_feedback


_transient = is_transient


def program(*, client, messages, query, path, config, budget, stopped, on_event, sample_id=0, temperature=0, tables=None):
    history = [dict(m) for m in messages]
    spent, sample_attempts, candidate, failure = 0.0, [], None, None
    # Keep the fallback local: voting samples share a client across threads.
    reasoning_override = None
    labels = []
    if config.get('sanity_retry'):
        if tables is None:
            from .parser import parse_workbook
            tables = parse_workbook(path, config)
        labels = table_labels(tables)
    execution_attempts, transient_retries = 0, 0
    while execution_attempts < config['max_attempts']:
        if stopped.is_set() or spent >= budget: break
        expense, metadata, charged = 0.0, {}, False
        try:
            kwargs = {'max_cost': budget-spent}
            if reasoning_override is not None: kwargs['reasoning'] = reasoning_override
            if temperature > 0: kwargs['temperature'] = temperature
            if hasattr(stopped, 'remaining') and getattr(client, 'supports_deadlines', False):
                kwargs['remaining_time'] = stopped.remaining()
            completion = client.complete(history, **kwargs)
            execution_attempts += 1
            code, expense = completion[:2]
            metadata = completion[2] if len(completion) > 2 else {}
            if metadata.get('finish_reason') == 'length':
                reasoning_override = config.get('length_fallback_reasoning', 'off')
            spent += expense
            charged = True
            if spent > budget:
                result = {'error': 'Question sample budget exceeded'}
            elif not code.strip():
                reason = metadata.get('finish_reason')
                result = {'error': 'Empty answer' + (f' (finish_reason={reason})' if reason else '')}
            else:
                if stopped.is_set():
                    raise TimeoutError('Question deadline reached')
                timeout = min(config['program_timeout'], stopped.remaining()) if hasattr(stopped, 'remaining') else config['program_timeout']
                result = run_program(code, path, timeout=timeout, config=config)
            failure = result.get('error')
            if failure and result.get('candidates'):
                failure += f". Candidates: {result['candidates']!r}"
            if not failure:
                try:
                    raw = result.get('answer')
                    if raw is None: raise ValueError('Empty answer')
                    candidate = format_answer(raw, query['query'])
                    if not candidate: failure = 'Empty answer'
                    elif _aggregate_label(candidate, query['query']):
                        failure = ('Answer is an aggregate row/column label (' + candidate + '); the question '
                                   'asks for a specific category, so exclude Total/All rows and recompute.')
                        candidate = None
                except Exception as exc:
                    failure = f'{type(exc).__name__}: {exc}'
            sample_attempts.append({'sample': sample_id, 'code': code, 'result': result,
                                    'cost': expense, 'usage': metadata, 'error': failure})
            on_event("observation", {"id": query["id"], "sample": sample_id, "result": result, "cost": expense})
            if not failure:
                if config.get('sanity_retry'):
                    issues = check_answer(candidate, query['query'], labels, code, raw)
                    sample_attempts[-1]['sanity'] = issues
                    on_event('sanity', {'id': query['id'], 'sample': sample_id,
                                        'answer': candidate, 'issues': issues, 'retry': False})
                    retry_history = history + [{'role': 'assistant', 'content': code}]
                    for retry in range(config.get('sanity_max_retries', 1)):
                        if not issues or stopped.is_set() or spent >= budget: break
                        retry_history.append({'role': 'user', 'content': retry_feedback(issues)})
                        corrected, extra_cost, extra, _ = program(
                            client=client, messages=retry_history, query=query, path=path,
                            config=dict(config, sanity_retry=False, max_attempts=1),
                            budget=budget-spent, stopped=stopped, on_event=on_event,
                            sample_id=sample_id, temperature=temperature)
                        spent += extra_cost
                        corrected_issues = []
                        if corrected is not None:
                            successful = extra[-1]
                            corrected_issues = check_answer(corrected, query['query'], labels,
                                                           successful['code'], successful['result'].get('answer'))
                        accepted = corrected is not None and not corrected_issues
                        for attempt in extra:
                            attempt.update(sanity_retry=True, sanity=corrected_issues,
                                           sanity_accepted=accepted)
                        sample_attempts.extend(extra)
                        on_event('sanity', {'id': query['id'], 'sample': sample_id,
                                            'answer': corrected, 'issues': corrected_issues,
                                            'retry': retry + 1, 'accepted': accepted})
                        if accepted:
                            candidate = corrected
                            break
                        if corrected is None: break
                        issues = corrected_issues
                        retry_history.append({'role': 'assistant', 'content': extra[-1]['code']})
                break
            candidate = None
            history.extend([{'role': 'assistant', 'content': code},
                            {'role': 'user', 'content': f'Execution failed: {failure}. Fix the program using exact labels and the listed candidates.'}])
        except Exception as exc:
            if is_credit_error(exc):
                stopped.set()
                on_event("fatal", {"error": str(exc), "reason": "Provider insufficient credit; further calls stopped"})
            if not charged:
                expense = float(getattr(exc, 'cost', 0.0))
                spent += expense
            metadata = getattr(exc, 'usage', metadata)
            failure = f'{type(exc).__name__}: {exc}'
            sample_attempts.append({'sample': sample_id, 'code': '', 'result': {'error': failure},
                                    'cost': expense, 'usage': metadata, 'error': failure})
            # Fatal (credit/stop) ends the sample; transient provider errors use the next attempt.
            if stopped.is_set() or is_credit_error(exc) or not _transient(exc):
                break
            if transient_retries >= config['max_transient_retries']:
                break
            transient_retries += 1
            # A stall already consumed its watchdog interval: retry immediately.
            delay = 0 if isinstance(exc, StallError) else min(float(config.get("transient_retry_sleep", 1.0)) * len(sample_attempts), 4.0)
            if hasattr(stopped, 'remaining'): delay = min(delay, stopped.remaining())
            time.sleep(delay)
    return candidate, spent, sample_attempts, failure



def _auxiliary(context, messages, allowance, kind, execute=False):
    """Exactly one bounded call; failures retain the current winner."""
    if allowance <= 0 or context['stopped'].is_set():
        return None, 0, []
    code, expense, metadata, result, error = '', 0, {}, {}, None
    try:
        kwargs = {'max_cost': allowance}
        if hasattr(context['stopped'], 'remaining') and getattr(context['client'], 'supports_deadlines', False):
            kwargs['remaining_time'] = context['stopped'].remaining()
        completion = context['client'].complete(messages, **kwargs)
        code, expense = completion[:2]
        metadata = completion[2] if len(completion) > 2 else {}
        if expense > allowance: raise ValueError('Auxiliary budget exceeded')
        if context['stopped'].is_set(): raise TimeoutError('Question deadline reached')
        if execute:
            if code.strip() == 'OK':
                result = {'verdict': 'OK'}
                candidate = None
            else:
                if '```python' not in code: raise ValueError('Expected OK or corrected python block')
                timeout = context['config']['program_timeout']
                if hasattr(context['stopped'], 'remaining'): timeout = min(timeout, context['stopped'].remaining())
                result = run_program(code, context['path'], timeout=timeout, config=context['config'])
                if result.get('error'): raise ValueError(result['error'])
                if result.get('answer') is None: raise ValueError('Empty corrected answer')
                candidate = format_answer(result['answer'], context['query']['query'])
        else:
            if not code.strip() or '```' in code: raise ValueError('Expected direct answer only')
            candidate = format_answer(code.strip(), context['query']['query'])
            result = {'answer': candidate}
        if candidate and _aggregate_label(candidate, context['query']['query']):
            raise ValueError('Aggregate label answer')
    except Exception as exc:
        expense = float(getattr(exc, 'cost', expense))
        metadata = getattr(exc, 'usage', metadata)
        candidate, error = None, f'{type(exc).__name__}: {exc}'
        if is_credit_error(exc):
            context['stopped'].set()
            context['on_event']('fatal', {'error': error, 'reason': 'Provider insufficient credit; further calls stopped'})
    attempt = {'kind': kind, 'code': code, 'result': result, 'cost': expense, 'usage': metadata, 'error': error}
    context['on_event'](kind, {'id': context['query']['id'], **attempt})
    return candidate, expense, [attempt]


def verify(answer, candidates, cost, attempts, context, allowance):
    if answer is None or not context['config'].get('verifier'): return answer, cost, attempts
    # Only a successful program for this winner can be checked.
    winning = next((a for a in attempts if a.get('sample') is not None and not a.get('error')
                    and format_answer(a['result'].get('answer'), context['query']['query']) == answer), None)
    if winning is None: return answer, cost, attempts
    messages = [dict(m) for m in context['messages']]
    messages[0]['content'] += '\nVerify the proposed solution. Return only OK or one corrected python code block defining answer.'
    messages.append({'role': 'user', 'content': f"Question: {context['query']['query']}\nProposed code:\n{winning['code']}\nFormatted answer: {answer}\nCheck labels, computation and formatting."})
    corrected, spent, extra = _auxiliary(context, messages, min(allowance, context['budget']-cost), 'verifier', True)
    # One correction is only one vote: an original/corrected tie retains original.
    if corrected is not None and corrected != answer and candidates.count(corrected) + 1 > candidates.count(answer):
        answer = corrected
    return answer, cost + spent, attempts + extra


def verified_program(**context):
    allowance = context['budget'] / 2 if context['config'].get('verifier') else 0
    result = program(**dict(context, budget=context['budget']-allowance))
    answer, cost, attempts = verify(result[0], [result[0]], result[1], result[2], context, allowance)
    return answer, cost, attempts, result[3]


def vote(**context):
    config = context['config']
    n = min(config['votes'], config.get('max_votes', config['votes'])) if config.get('vote_mode', 'fixed') == 'adaptive' else config['votes']
    slots = n + int(config.get('hybrid_text', False)) + int(config.get('verifier', False))
    allowance = context['budget'] / slots
    def sample(i):
        return program(**dict(context, budget=allowance, sample_id=i, temperature=.7 if n > 1 else 0))
    first = min(config.get('vote_first', 2), n) if config.get('vote_mode', 'fixed') == 'adaptive' else n
    futures = [launch(sample, i) for i in range(first)]
    results = [future.result() for future in futures]
    # vote_first=1: accept a single executable answer; otherwise require the first two to agree.
    agreed = results[0][0] is not None and all(r[0] == results[0][0] for r in results) if results else False
    if not agreed:
        for i in range(first, n):
            if context.get('stopped') is not None and context['stopped'].is_set(): break
            results.append(sample(i))
    candidates = [r[0] for r in results if r[0] is not None]
    answer = max(candidates, key=candidates.count) if candidates else None
    cost = sum(r[1] for r in results)
    attempts = [a for r in results for a in r[2]]
    if config.get('hybrid_text') and len(set(candidates)) > 1 and candidates.count(answer) <= len(candidates)/2:
        messages = [dict(m) for m in context['messages']]
        messages[0] = {'role': 'system', 'content': 'Answer the question directly using the supplied table. Return the answer only, in the requested format. Do not write code.'}
        direct, spent, extra = _auxiliary(context, messages, min(allowance, context['budget']-cost), 'hybrid_text')
        cost += spent
        attempts += extra
        if direct is not None:
            candidates.append(direct)
            # The textual answer decides a residual tie, including a new answer.
            answer = max(candidates, key=lambda c: (candidates.count(c), c == direct))
    answer, cost, attempts = verify(answer, candidates, cost, attempts, context, allowance)
    if context.get('on_event'):
        context['on_event']('voting', {'id': context['query']['id'], 'samples_used': sum(bool(r[2]) for r in results),
                                     'vote_mode': config.get('vote_mode', 'fixed'), 'candidates': candidates, 'answer': answer})
    return answer, cost, attempts, None if answer is not None else next((r[3] for r in results if r[3]), 'No answer within budget')


STRATEGIES = {'program': verified_program, 'vote': vote}


def _aggregate_label(answer, question):
    """Argmax/ranking answers must not be aggregate labels unless the question mentions them."""
    import re
    q = question.lower()
    words = [w.strip().lower() for w in answer.split(',')]
    agg = re.compile(r'^(total|all items|all industries|all)\b')
    return any(agg.match(w) for w in words) and not re.search(r'\b(total|all items|all industries)\b', q)
