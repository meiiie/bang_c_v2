import json
from types import SimpleNamespace
import openpyxl
import pandas as pd
import pytest
from solver.config import load_config, apply_overrides
from solver.helpers import HELPERS, render_helpers, helper_namespace, children, leaves, age_at, sheet, to_number
from solver.parser import STAGES, parse_workbook
from solver.program import build_prompt, run_program
from solver.pipeline import Pipeline, config_hash, input_hash
from solver.strategies import vote
from solver.formatter import format_answer
from solver.llm import LLMClient, InsufficientCredit


def test_config_merge(tmp_path, monkeypatch):
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'budget': .02, 'profiles': {'fast': {'max_tokens': 1000, 'votes': 2}}, 'retries': 1}))
    monkeypatch.setenv('SOLVER_MAX_TOKENS', '2000')
    config = load_config(path, 'fast')
    assert config['budget'] == .02 and config['max_tokens'] == 2000
    assert config['votes'] == 2 and config['workers'] == 8
    assert config['max_attempts'] == 2
    assert apply_overrides(config, ['max_tokens=3000', 'budget=0.04'])['max_tokens'] == 3000
    with pytest.raises(ValueError): apply_overrides(config, ['model=other'])
    with pytest.raises(ValueError): apply_overrides(config, ['budget=0.11'])
    with pytest.raises(ValueError): apply_overrides(config, ['unknown=1'])
    with pytest.raises(ValueError): load_config(path, 'missing')


def test_helpers_registry_rendering_and_injection(data_dir):
    names = {spec.name for spec in HELPERS}
    assert set(helper_namespace()) == names
    for spec in HELPERS:
        assert spec.signature in render_helpers()
        assert helper_namespace()[spec.name] is spec.fn
    config = load_config()
    config['helpers'] = ['to_number', 'sheet']
    prompt = build_prompt('score', parse_workbook(data_dir/'xlsx_files/p4_t01.xlsx'), config=config)[0]['content']
    assert 'to_number(x)' in prompt and 'find_row(df' not in prompt
    path = data_dir/'xlsx_files/p4_t01.xlsx'
    assert run_program("answer=to_number('1,234')", path, config=config) == {'answer': 1234}
    assert run_program("answer=sheet(tables, '1')['Score'].max()", path, config=config) == {'answer': 99}
    assert 'NameError' in run_program("answer=find_col(tables[0], 'Score')", path, config=config)['error']


def test_generic_helpers():
    df = pd.DataFrame({'birth': ['2000-01-01']*4, 'event': ['2000-01-11']*4}, index=['A', 'A > B', 'A > B > C', 'A > D'])
    assert list(children(df, 'A').index) == ['A > B', 'A > D']
    assert leaves(df) == ['A > B > C', 'A > D']
    assert leaves(['A', 'A > B', 'X']) == ['A > B', 'X']
    assert age_at(df, 'birth', 'event').tolist() == [10]*4
    assert age_at(df, 'birth', '2000-01-02').tolist() == [1]*4
    assert to_number('(1,234)') == -1234


def test_stage_registry_order_and_optional_stage(data_dir, monkeypatch):
    expected = ['load_grid', 'unmerge_fill', 'drop_titles_footnotes', 'normalize_values', 'split_tables', 'detect_headers', 'build_row_paths', 'clean_labels', 'enrichers']
    assert [fn.__name__ for fn in STAGES] == expected
    config = load_config()
    assert config['stages'] == expected and config['enrichers'] == []
    observed = []
    def observe(state):
        observed.append(len(state['tables']))
        return state
    monkeypatch.setattr('solver.parser.STAGES', STAGES + [observe])
    config['stages'] += ['observe']
    assert parse_workbook(data_dir/'xlsx_files/p4_t01.xlsx', config)
    assert observed == [1]
    config['stages'].remove('observe')
    parse_workbook(data_dir/'xlsx_files/p4_t01.xlsx', config)
    assert observed == [1]


def test_manifest_events_and_hashes(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    events = []
    p = Pipeline(None, on_event=lambda kind, data: events.append(kind))
    q = {'id': 'q', 'file_name': 'p4_t01.xlsx', 'query': 'Score?'}
    assert p.solve_all([q], data_dir, mode='dump', dump_dir=tmp_path/'prompts') == [{'id': 'q', 'answer': ''}]
    manifest = json.loads((p.run_dir/'manifest.json').read_text())
    assert manifest['config_hash'] == config_hash(p.config)
    assert manifest['input_hash'] == input_hash(data_dir, [q])
    assert manifest['model'] == 'qwen/qwen3.7-flash'
    assert 'git_rev' in manifest
    assert events == ['run_started', 'parsed', 'question', 'run_finished']
    assert [json.loads(line)['kind'] for line in p.trace_path.read_text().splitlines()] == events
    config = dict(p.config, max_tokens=5000)
    assert config_hash(config) != manifest['config_hash']


def test_prompt_all_sheets_and_priority(tmp_path):
    w = openpyxl.Workbook()
    w.active.title = 'Unrelated'
    w.active.append(['Name', 'Value'])
    for i in range(60): w.active.append([f'row{i}', i])
    s = w.create_sheet('Revenue')
    s.append(['Category', 'Revenue'])
    s.append(['Sales', 42])
    path = tmp_path/'sheets.xlsx'; w.save(path)
    tables = parse_workbook(path)
    config = load_config(); config['prompt_char_budget'] = 3500
    messages = build_prompt('Revenue Sales?', tables, config=config)
    user = messages[1]['content']
    assert sum(len(m['content']) for m in messages) <= 3500
    assert 'tables[0] Sheet: Unrelated' in user and 'tables[1] Sheet: Revenue' in user
    assert user.index('Sheet: Revenue') < user.index('Sheet: Unrelated')
    assert '42' in user


@pytest.mark.parametrize('query,expected', [
    ('Which Food category has the most?', 'Food at home'),
    ('Choose among: Food, Clothing, and Housing.', 'Food'),
    ('Choose among: Food, Food at home, and Housing.', 'Food at home'),
    ('What about Food and Clothing?', 'Food at home'),
])
def test_candidate_mapping(query, expected):
    assert format_answer('Food at home', query) == expected


def test_credit_error_stops_whole_run(data_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    class CreditError(RuntimeError): status_code = 402
    class Client:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise CreditError('Insufficient credits')
    c = Client(); config = load_config(); config.update(strategy='program', workers=1)
    p = Pipeline(c, config=config)
    qs = [{'id': f'q{i}', 'file_name': 'p4_t01.xlsx', 'query': 'score'} for i in range(4)]
    assert all(r['answer'] == '0' for r in p.solve_all(qs, data_dir))
    assert c.calls == 1 and p.stopped.is_set()
    assert 'insufficient credit' in capsys.readouterr().err
    assert '"kind": "fatal"' in p.trace_path.read_text()


def test_provider_credit_latch(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test')
    c = LLMClient()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('insufficient credits')
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    for _ in range(2):
        with pytest.raises(InsufficientCredit): c.complete([{'content': 'test'}])
    assert len(calls) == 1


def test_vote_tie_prefers_first_success(monkeypatch):
    def stub(**ctx):
        i = ctx['sample_id']
        return [None, 'A', 'B'][i], 0, [], None
    monkeypatch.setattr('solver.strategies.program', stub)
    assert vote(config={'votes': 3}, budget=.03)[0] == 'A'


def test_input_hash_tracks_content_and_names(tmp_path):
    (tmp_path/'xlsx_files').mkdir()
    (tmp_path/'queries.json').write_text('[{"file_name":"a.xlsx"}]')
    workbook = tmp_path/'xlsx_files/a.xlsx'
    workbook.write_bytes(b'first')
    initial = input_hash(tmp_path)
    workbook.write_bytes(b'second')
    changed = input_hash(tmp_path)
    assert initial != changed
    workbook.rename(tmp_path/'xlsx_files/b.xlsx')
    (tmp_path/'queries.json').write_text('[{"file_name":"b.xlsx"}]')
    assert input_hash(tmp_path) != changed
