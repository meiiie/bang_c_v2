"""Configurable harness, per-run provenance, events and ordered CSV results."""
from .config import launch, DeadlineStop, GuardedClient
from .config import fallback_answer, initial_rows, normalize_query_ids, write_csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from .config import load_config, validate_config
from .parser import parse_workbook
from .program import build_prompt
from .strategies import STRATEGIES
from .checkpoint import Checkpoint, answer_config_hash, answer_key, file_hash


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def workbook_path(input_dir, file_name):
    """Resolve a query's workbook without permitting other input files."""
    if (not isinstance(file_name, str) or not file_name
            or any(c in file_name for c in ('/', '\\', ':', '\0'))
            or Path(file_name).suffix.lower() not in {'.xlsx', '.xlsm', '.xls'}):
        raise ValueError('Invalid workbook filename')
    directory = Path(input_dir).resolve() / 'xlsx_files'
    path = (directory / file_name).resolve()
    if (directory.resolve() != directory or path.parent != directory
            or path.suffix.lower() not in {'.xlsx', '.xlsm', '.xls'}):
        raise ValueError('Invalid workbook filename')
    return path


def input_hash(input_dir, queries=None):
    root = Path(input_dir)
    digest = hashlib.sha256()
    query_bytes = (root / 'queries.json').read_bytes()
    if queries is None:
        queries = json.loads(query_bytes.decode('utf-8-sig'))
    digest.update(b'queries.json\0' + query_bytes + b'\0')
    # Never enumerate the input folder, even for provenance/checkpoint hashes.
    paths = {}
    for query in queries:
        try:
            paths[query['file_name']] = workbook_path(root, query['file_name'])
        except (OSError, ValueError):
            continue  # solve() records invalid/missing workbooks per question.
    for name, path in sorted(paths.items()):
        digest.update(('xlsx_files/' + name).encode())
        digest.update(b'\0')
        try:
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        except OSError:
            digest.update(b'missing')
        digest.update(b'\0')
    return digest.hexdigest()


class Pipeline:
    def __init__(self, client, budget=None, retries=None, log_path=None, self_consistency=False, votes=None, config=None, on_event=None):
        self.config = dict(load_config() if config is None else config)
        if budget is not None: self.config['budget'] = budget
        if retries is not None:
            self.config['retries'] = retries
            self.config['max_attempts'] = retries + 1
        if votes is not None: self.config['votes'] = votes
        if self_consistency or votes is not None: self.config['strategy'] = 'vote'
        validate_config(self.config)
        if self.config['strategy'] not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.config['strategy']}")
        self.client = client
        self.budget, self.votes = self.config['budget'], self.config['votes']
        self.retries = self.config['max_attempts'] - 1
        self.run_dir = Path('runs')/str(time.time_ns())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path) if log_path else self.run_dir/'questions.jsonl'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir/'trace.jsonl'
        self.trace_path.touch()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.deadline = time.monotonic() + self.config['total_time_limit']
        self.event_hook = on_event
        self.checkpoint = None
        self.checkpoint_keys = {}
        self.question_records = {}
        self._manifest(None)

    def _manifest(self, input_dir, queries=None):
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent.parent,
                                      capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            revision = None
        manifest = {'schema_version': 1, 'config': self.config, 'config_hash': config_hash(self.config),
                    'input_hash': input_hash(input_dir, queries) if input_dir else None,
                    'model': self.config['model'], 'git_rev': revision}
        (self.run_dir/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    def on_event(self, kind, data):
        with self.lock:
            with self.trace_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'kind': kind, 'data': data, 'timestamp': time.time()}, ensure_ascii=False)+'\n')
        if kind == 'fatal':
            print(f"Run stopped: {data['reason']}: {data['error']}", file=sys.stderr)
        if self.event_hook:
            self.event_hook(kind, data)

    def solve(self, query, input_dir, mode='llm', dump_dir='prompts', deadline=None):
        query = normalize_query_ids([query])[0]
        with self.lock:
            previous = self.question_records.get(str(query['id']), {})
        prior_cost = previous.get('total_cost', 0.0)
        cost, attempts, answer, error = 0.0, [], '', None
        started = time.monotonic()
        deadline = min(deadline or (started + self.config['question_timeout']), self.deadline - self.config['time_guard_margin'])
        stop = DeadlineStop(self.stopped, deadline)
        # Direct solve remains supported and gets the same provenance as solve_all.
        if not getattr(self, '_input_dir', None):
            with self.lock:
                if not getattr(self, '_input_dir', None):
                    self._manifest(input_dir, [query])
                    self._input_dir = str(input_dir)
        try:
            if stop.is_set():
                raise TimeoutError('Run stopped or time budget exhausted')
            path = workbook_path(input_dir, query['file_name'])
            if not query['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in query['id']): raise ValueError('Invalid query ID')
            tables = parse_workbook(path, self.config)
            self.on_event('parsed', {'id': query['id'], 'tables': [t.name for t in tables]})
            from .fast_path import lookup
            direct = lookup(query['query'], tables) if self.config.get('fast_path') and mode != 'dump' else None
            messages = build_prompt(query['query'], tables, query['id'], self.config) if direct is None else []
            if mode == 'dump':
                dest = Path(dump_dir)
                dest.mkdir(parents=True, exist_ok=True)
                (dest/f"{query['id']}.txt").write_text('\n\n'.join(f"{m['role']}:\n{m['content']}" for m in messages), encoding='utf-8')
            elif direct is not None:
                answer = direct
                self.on_event('fast_path', {'id': query['id'], 'answer': answer})
            else:
                answer, cost, attempts, error = STRATEGIES[self.config['strategy']](
                    client=GuardedClient(self.client, stop), messages=messages, query=query, path=path, config=self.config,
                    budget=max(0, self.budget-prior_cost), stopped=stop, on_event=self.on_event, tables=tables)
                if answer is None:
                    answer, error = fallback_answer(query), error or 'No answer within budget'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            answer = '' if mode == 'dump' else fallback_answer(query)
        record = {'id': query['id'], 'mode': mode, 'answer': answer, 'error': error,
                  'samples_used': len({a['sample'] for a in attempts if 'sample' in a}),
                  'cost': cost, 'total_cost': prior_cost + cost, 'seconds': time.monotonic()-started, 'attempts': attempts}
        with self.lock:
            self.question_records[str(query['id'])] = record
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
            key = self.checkpoint_keys.get(str(query['id']))
            if mode == 'llm' and self.checkpoint is not None and key and not error and answer and not stop.is_set():
                self.checkpoint.append(key, {'id': query['id'], 'answer': answer})
        self.on_event('question', record)
        return {'id': query['id'], 'answer': answer}

    def solve_all(self, queries, input_dir, workers=None, mode='llm', dump_dir='prompts', on_answer=None,
                  output_path=None, checkpoint=True, fresh=False, client_factory=None):
        queries = normalize_query_ids(queries)
        workers = self.config['workers'] if workers is None else workers
        if workers < 1: raise ValueError('workers must be positive')
        self._manifest(input_dir, queries)
        self._input_dir = str(input_dir)
        self.on_event('run_started', {'mode': mode, 'questions': len(queries)})
        rows = initial_rows(queries, mode)
        positions = {row['id']: i for i, row in enumerate(rows)}
        self.checkpoint, self.checkpoint_keys = None, {}
        remaining = list(queries)
        checkpoint_path = self.config.get('checkpoint_path') or (str(output_path) + '.checkpoint.jsonl' if output_path else None)
        if mode == 'llm' and checkpoint and checkpoint_path:
            self.checkpoint = Checkpoint(checkpoint_path, fresh=fresh)
            settings_hash = answer_config_hash(self.config)
            workbook_hashes, remaining, resumed = {}, [], set()
            for query in queries:
                name = query['file_name']
                if name not in workbook_hashes:
                    try:
                        workbook_hashes[name] = file_hash(workbook_path(input_dir, name))
                    except (OSError, ValueError):
                        workbook_hashes[name] = None
                key = answer_key(query, workbook_hashes[name], self.config['model'], settings_hash) if workbook_hashes[name] else None
                if key:
                    self.checkpoint_keys[str(query['id'])] = key
                cached = self.checkpoint.answers.get(key)
                if cached and cached['id'] == str(query['id']):
                    row = {'id': str(query['id']), 'answer': cached['answer']}
                    rows[positions[row['id']]] = row
                    if on_answer: on_answer(row)
                    resumed.add(row['id'])
                else:
                    remaining.append(query)
            self.on_event('checkpoint_resume', {'resumed_ids': len(resumed), 'pending': len(remaining),
                                                'path': str(checkpoint_path), 'fresh': fresh})
        pending = iter(remaining)
        def publish(row):
            rows[positions[str(row['id'])]] = row
            if output_path:
                write_csv(output_path, rows)
            if on_answer:
                on_answer(row)

        if output_path:
            write_csv(output_path, rows)
        # Restore durable answers before initializing the provider.
        if remaining and client_factory is not None:
            self.client = client_factory()
        abandoned = {}
        active, retired, exhausted = {}, 0, False
        replacements_pending = 0
        max_abandoned = self.config['max_abandoned_workers']
        cutoff = self.deadline - self.config['time_guard_margin']
        while True:
            now = time.monotonic()
            if self.stopped.is_set() or now >= cutoff:
                break
            while not exhausted and len(active) < workers and retired <= max_abandoned:
                if self.stopped.is_set() or time.monotonic() >= cutoff:
                    break
                try:
                    query = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                deadline = min(time.monotonic() + self.config['question_timeout'], cutoff)
                future = launch(self.solve, query, input_dir, mode, dump_dir, deadline)
                active[future] = (query, deadline)
                if replacements_pending:
                    replacements_pending -= 1
                    self.on_event('worker_replaced', {'id': query['id'], 'abandoned_workers': retired})
            for future, (query, deadline) in list(active.items()):
                if future.done():
                    try:
                        row = future.result()
                    except BaseException as exc:
                        row = rows[positions[str(query['id'])]]
                        self.on_event('question_failed', {'id': query['id'], 'error': str(exc)})
                    publish(row)
                    del active[future]
                elif time.monotonic() >= deadline:
                    # Abandon the daemon thread, but replace its scheduling capacity
                    # while within the cap. Rescue may recover its finished answer.
                    abandoned[str(query['id'])] = future
                    retired += 1
                    replacements_pending += 1
                    del active[future]
                    self.on_event('question_timeout', {'id': query['id']})
            if not active and (exhausted or retired > max_abandoned):
                break
            time.sleep(.01)
        # Rescue is sequential, but still uses a daemon future so a broken
        # transport cannot hold the global guard or shutdown open.
        if mode == 'llm' and self.config['rescue_pass'] and retired <= max_abandoned:
            for query in remaining:
                identity = str(query['id'])
                if self.stopped.is_set() or time.monotonic() >= cutoff:
                    break
                if identity in abandoned and not abandoned[identity].done():
                    # Its billable calls may still be running; do not double-spend.
                    continue
                with self.lock:
                    record = self.question_records.get(identity)
                if record and not record['error'] and record['answer']:
                    if identity in abandoned:
                        # The main scheduler discarded this future while it was
                        # still running. Its completed answer needs publication
                        # and may have missed the original checkpoint deadline.
                        with self.lock:
                            key = self.checkpoint_keys.get(identity)
                            if (self.checkpoint is not None and key and
                                    time.monotonic() < cutoff and not self.stopped.is_set()):
                                cached = self.checkpoint.answers.get(key)
                                if not cached or cached['answer'] != record['answer']:
                                    self.checkpoint.append(key, {'id': identity, 'answer': record['answer']})
                        publish({'id': identity, 'answer': record['answer']})
                        self.on_event('rescue_recovered', {'id': identity})
                    continue
                if record and record['total_cost'] >= self.budget:
                    continue
                deadline = min(time.monotonic() + self.config['question_timeout'], cutoff)
                self.on_event('rescue_started', {'id': query['id']})
                future = launch(self.solve, query, input_dir, mode, dump_dir, deadline)
                while not future.done() and not self.stopped.is_set() and time.monotonic() < deadline:
                    time.sleep(min(.01, max(0, deadline-time.monotonic())))
                if future.done():
                    try:
                        publish(future.result())
                    except BaseException as exc:
                        self.on_event('question_failed', {'id': query['id'], 'error': str(exc)})
                else:
                    retired += 1
                    self.on_event('question_timeout', {'id': query['id'], 'rescue': True})
                    if retired > max_abandoned:
                        break
        self.on_event('run_finished', {'questions': len(rows), 'stopped': self.stopped.is_set(),
                                       'timed_out_workers': retired})
        return rows
