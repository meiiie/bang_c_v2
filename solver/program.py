"""Compact prompts and resource-limited isolated program execution."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import tempfile

_EXECUTION_SLOTS = {}
_EXECUTION_LOCK = threading.Lock()


def build_prompt(query, tables, query_id='', config=None):
    from solver.config import load_config
    from solver.helpers import render_helpers
    config = load_config() if config is None else config
    system = ('Compute the answer using Python. Return one python code block defining answer. '
              'Preloaded tables is a list of pandas DataFrames; pd, math, datetime are available. '
              'Columns are exact strings in a Python list with numbered zero-based positions. '
              'Rows are the DataFrame index, NOT a column: use df.loc["exact index string"]. '
              'No file, network, process access. Imports only pandas, math, datetime. '
              'For a sum of named items use reported totals; do not add a total to its own parts. When a question asks how many of the listed/named categories meet a condition, count those categories (rows), not the values in them. '
              'Use exactly the rows/columns/reference dates the question states. '
              'If a question asks about age/youngest/oldest relative to a date in the table '
              '(e.g. on their admission date), use age_at with that column; otherwise use birth date. '
              'For averages across a group, average the categories the question refers to as displayed in the table. For argmax/ranking among categories, exclude aggregate rows whose '
              'last label segment starts with Total unless requested. Return names exactly as the '
              'row/column label last path segment with footnote marks removed. Return scalar or list; '
              'keep values in the units displayed in the table (never multiply by notes such as "in thousands") unless the question explicitly asks to convert; '
              'do not round or format; return full precision and leave final formatting to the caller. All table rows remain available at execution.')
    if config.get('average_policy', 'question') == 'all_displayed':
        system += ('\nAverage policy: for "average across all X", take the plain mean of every value '
                   'displayed for X in that row/column, including subtotal columns, unless the question '
                   'restricts the set.')
    system += '\nHelpers:\n' + render_helpers(config['helpers'])
    # Keep numeric-heavy displays small too. Relevant rows come first when
    # necessary; displayed row numbers always retain their original iloc position.
    prefix = f'Query ID: {query_id}\nQuestion: {query}\n'
    remaining = config['prompt_char_budget'] - len(system) - len(prefix)
    words = set(re.findall(r'[^\W\d_]{3,}', query.casefold())) - {'the', 'and', 'for', 'with', 'answer', 'what', 'only'}
    def score(text):
        return len(words & set(re.findall(r'[^\W\d_]{3,}', text.casefold())))
    order = sorted(range(len(tables)), key=lambda i: (-score(tables[i].name + ' ' + ' '.join(tables[i].header_paths) + ' '.join(r.label_path for r in tables[i].rows)), i))
    # Reserve an identity for every table, even when its detailed display is omitted.
    identities = [f'tables[{i}] Sheet: {tables[i].name.rsplit(":", 1)[0]}; Table: {tables[i].name}' for i in order]
    if remaining < sum(len(x) + 2 for x in identities):
        raise ValueError('Prompt char budget too small to include all table identities')
    remaining -= sum(len(x) + 3 for x in identities)
    chunks = []
    for pos, i in enumerate(order):
        table = tables[i]
        weights = [1 + score(tables[j].name + ' ' + ' '.join(tables[j].header_paths)) for j in order[pos:]]
        limit = max(0, int(remaining * weights[0] / sum(weights)))
        indices = sorted(range(len(table.rows)), key=lambda j: (-score(table.rows[j].label_path), j)) if len(table.to_text(10**7, serialization=config.get('serialization', 'pipe'))) > limit else None
        rendered = table.to_text(limit, row_indices=indices, serialization=config.get('serialization', 'pipe'))
        chunks.append(identities[pos] + '\n' + rendered)
        remaining -= len(rendered)
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': prefix + '\n\n'.join(chunks)}]


# Compatibility imports for callers of the original program module.
if __name__ != "__main__":
    from solver.helpers import find_row, find_col


def extract_code(code):
    match = re.search(r'```(?:python)?\s*\n(.*?)```', code, re.S)
    return match.group(1) if match else code.strip()


def validate_code(code):
    tree = ast.parse(code)
    safe_conversions = {'to_datetime', 'to_numeric', 'to_timedelta', 'to_numpy', 'to_list', 'to_dict', 'to_period', 'to_timestamp'}
    forbidden = {'open', 'exec', 'eval', 'compile', 'getattr', 'setattr', 'delattr',
                 'globals', 'locals', 'vars', '__import__', 'input', 'breakpoint',
                 'ExcelFile', 'ExcelWriter', 'HDFStore', 'attrgetter', 'methodcaller',
                 'modules', 'gi_frame', 'ag_frame', 'cr_frame', 'tb_frame',
                 'f_globals', 'f_builtins', 'f_locals', 'f_back', 'import_module',
                 'import_optional_dependency', 'getmembers', 'getmembers_static',
                 'getattr_static', 'currentframe'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(m not in ('pandas', 'math', 'datetime') for m in modules):
                raise ValueError('Only pandas, math, datetime imports allowed')
            if isinstance(node, ast.ImportFrom) and any(a.name == '*' or a.name.startswith(('read_', '_')) or a.name in forbidden for a in node.names):
                raise ValueError('Forbidden imported function')
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith(('__', 'read_'))):
            raise ValueError(f'Forbidden name: {node.id}')
        # Public pandas objects expose modules such as os. Do not let generated
        # code monkeypatch the audit hook's dependencies through those objects.
        # DataFrame item/loc assignments remain available as Subscript targets.
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ValueError('Forbidden attribute assignment or deletion')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr.startswith('read_') or
               (node.attr.startswith('to_') and node.attr not in safe_conversions) or
               node.attr in forbidden | {'save', 'load', 'dump', 'loads', 'dumps', 'query'}):
            raise ValueError(f'Forbidden attribute: {node.attr}')
    return tree


def run_program(code, workbook_path, timeout=20, config=None):
    try:
        from solver.config import load_config
        config = load_config() if config is None else config
        timeout = min(timeout, config['program_timeout'], 20)
        code = extract_code(code)
        validate_code(code)
        env = dict(os.environ)
        for key in list(env):
            if 'KEY' in key or 'TOKEN' in key: env.pop(key)
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with _EXECUTION_LOCK:
            slots = _EXECUTION_SLOTS.setdefault(config['program_workers'], threading.BoundedSemaphore(config['program_workers']))
        import time
        started = time.monotonic()
        if not slots.acquire(timeout=timeout):
            return {'error': 'Execution slot deadline exceeded'}
        try:
            timeout -= time.monotonic() - started
            if timeout <= 0:
                return {'error': 'Execution deadline exceeded'}
            path = str(Path(workbook_path).resolve())
            with tempfile.TemporaryDirectory(prefix='excel-qa-') as cwd:
                child = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()),
                                        path, json.dumps(config)],
                                       input=code, text=True, capture_output=True, timeout=timeout,
                                       env=env, cwd=cwd)
        finally:
            slots.release()
        if child.returncode:
            return {'error': (child.stderr or 'Worker failed')[-3000:]}
        return json.loads(child.stdout)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def _install_file_guard(workbook_path):
    """Defense in depth for Python file readers, in addition to AST restrictions.

    Library reads allow lazy imports and package data. This is a lightweight
    leakage guard, not an OS security boundary for hostile native code.
    """
    import sysconfig
    workbook = os.path.realpath(workbook_path)
    libraries = tuple({os.path.realpath(sysconfig.get_path(key))
                       for key in ('stdlib', 'platstdlib', 'purelib', 'platlib')})
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    mutations = {'os.rename', 'os.remove', 'os.link', 'os.symlink', 'os.truncate',
                 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.chown', 'os.utime'}

    def audit(event, args):
        # A spawned shell/native process would bypass Python's open audit hook.
        if event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn',
                     'os.fork', 'os.forkpty', 'os.spawn'}:
            raise PermissionError('Process access disabled in program worker')
        # Prevent replacing the allowlisted workbook with a forbidden file.
        if event in mutations:
            raise PermissionError('File changes disabled in program worker')
        if event in {'sys.settrace', 'sys.setprofile', 'pickle.find_class'}:
            raise PermissionError('Runtime introspection disabled in program worker')
        if event != 'open':
            return
        path, mode, flags = args
        if flags & write_flags or (isinstance(mode, str) and any(c in mode for c in 'wax+')):
            raise PermissionError('File writes disabled in program worker')
        # Do not permit reopening arbitrary inherited descriptors.
        if isinstance(path, int):
            raise PermissionError('File descriptor access disabled in program worker')
        resolved = os.path.realpath(os.fsdecode(path))
        if resolved != workbook and not any(
                resolved.startswith(root + os.sep) for root in libraries):
            raise PermissionError('File access restricted to current workbook and Python libraries')

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)


def _worker():
    import inspect
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (700*1024**2, 700*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pandas as pd
    import math
    import datetime
    from solver.parser import parse_workbook
    import builtins
    code = sys.stdin.read()
    validate_code(code)
    allowed = {k: getattr(builtins, k) for k in ('abs','all','any','bool','dict','enumerate','float','int','isinstance','len','list','max','min','range','reversed','round','set','sorted','str','sum','tuple','zip','ValueError','TypeError','Exception')}
    def safe_print(*args, **kwargs):
        kwargs['file'] = sys.stderr
        builtins.print(*args, **kwargs)
    allowed['print'] = safe_print
    def safe_import(name, *args, **kwargs):
        if name not in ('pandas', 'math', 'datetime'): raise ImportError(name)
        return {'pandas': pd, 'math': math, 'datetime': datetime}[name]
    allowed['__import__'] = safe_import
    from solver.helpers import helper_namespace
    # Load solver configuration code before restricting reads to the workbook
    # and installed Python libraries (the source checkout is not allowlisted).
    from solver import config as _config
    config = json.loads(sys.argv[2])
    _install_file_guard(sys.argv[1])
    scope = {'tables': [t.to_dataframe() for t in parse_workbook(sys.argv[1], config)],
             'pd': pd, 'math': math, 'datetime': datetime,
             '__builtins__': allowed}
    scope.update(helper_namespace(config['helpers']))
    def _round_half_up(x, ndigits=None):
        import decimal
        try:
            q = decimal.Decimal(1) if not ndigits else decimal.Decimal(1).scaleb(-ndigits)
            r = decimal.Decimal(repr(float(x))).quantize(q, rounding=decimal.ROUND_HALF_UP)
            return int(r) if ndigits is None else float(r)
        except Exception:
            return __builtins__['round'](x, ndigits) if isinstance(__builtins__, dict) else __import__('builtins').round(x, ndigits)
    scope['round'] = _round_half_up
    import socket
    def no_network(*args, **kwargs):
        raise PermissionError('Network disabled in program worker')
    socket.socket = no_network
    socket.create_connection = no_network
    try:
        exec(compile(code, '<answer>', 'exec'), scope)
        answer = scope.get('answer')
        if answer is None:
            for name in ('solve', 'main', 'compute', 'get_answer'):
                candidate = scope.get(name)
                if callable(candidate) and getattr(candidate, '__globals__', None) is scope:
                    answer = candidate
                    break
        if callable(answer):
            try:
                inspect.signature(answer).bind(scope['tables'])
            except TypeError:
                answer = answer()
            else:
                answer = answer(scope['tables'])
    except KeyError as exc:
        # Direct pandas selections get the same useful feedback as lookup helpers.
        from difflib import get_close_matches
        missing = str(exc.args[0]) if exc.args else ''
        candidates = []
        for frame in scope['tables']:
            labels = {}
            for kind, values in (('index', frame.index), ('columns', frame.columns)):
                values = list(values)
                labels[kind] = get_close_matches(missing, values, n=8, cutoff=0.25) or values
            candidates.append(labels)
        print(json.dumps({'error': f'KeyError: {exc}', 'candidates': candidates}))
        return
    def normalize(v):
        if isinstance(v, (list, tuple)): return [normalize(x) for x in v]
        if hasattr(v, 'item'): return v.item()
        if isinstance(v, (datetime.date, datetime.datetime)): return v.isoformat()
        return v
    print(json.dumps({'answer': normalize(answer)}, allow_nan=False))


if __name__ == '__main__':
    try: _worker()
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
