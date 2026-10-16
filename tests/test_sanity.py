import json
import threading
import unicodedata

import pytest

from solver.config import load_config, apply_overrides
from solver.parser import Row, Table
from solver.sanity import (check_answer, closest_labels, expected_item_count,
                           missing_program_labels, normalize_label, table_labels)
from solver.strategies import program


LABELS = ['Region > MIỀN NAM', 'MIỀN NAM', 'Cedar', 'Industry', 'Score']


@pytest.mark.parametrize('question', [
    'How many workers?', 'Count the workers.', 'What is the total?',
    'Find the sum.', 'What is the number of workers?', 'Find the combined value.',
])
@pytest.mark.parametrize('answer,raw', [('0', 0), ('0.00', 0.0), ('0, 0', [0, 0])])
def test_zero_counts(question, answer, raw):
    issues = check_answer(answer, question, LABELS, raw=raw)
    assert len(issues) == 1 and 'exactly zero' in issues[0]['reason']


@pytest.mark.parametrize('answer,question,raw', [
    ('0', 'What is the change?', 0), ('1', 'How many?', 1),
    ('0, 1', 'What are the combined counts?', [0, 1]),
    ('No', 'Is the total zero? Answer Yes or No.', False),
])
def test_zero_does_not_flag_other_answers(answer, question, raw):
    assert not check_answer(answer, question, LABELS, raw=raw)


def test_checks_formatted_list_values():
    assert check_answer('0, 0', 'Give the combined counts as two integers.',
                        LABELS, raw=[0.1, 0.2])
    assert not check_answer('Cedar, Industry', 'Which two names?', LABELS,
                            raw=['Group > Cedar', 'Group > Industry'])


@pytest.mark.parametrize('question,count', [
    ('Return three sector names.', 3), ('Answer with two integers.', 2),
    ('Return 4 names.', 4), ('Answer as `a, b`.', 2),
    ('What happened in 2020?', None), ('Round to two decimal places.', None),
])
def test_expected_count(question, count):
    assert expected_item_count(question) == count


def test_list_arity():
    issues = check_answer('Cedar, Industry', 'Return three sector names.',
                          LABELS, raw=['Cedar', 'Industry'])
    assert 'requests 3 items' in issues[0]['reason']
    assert not check_answer('1, 2', 'Answer as `a, b`.', LABELS, raw=[1, 2])
    assert 'has 1' in check_answer('1', 'Return two integers.', LABELS, raw=1)[0]['reason']
    # A single label with a comma remains one item when returned as a list.
    assert not check_answer('Total, all ages, Industry', 'Return two names.',
                            LABELS, raw=['Total, all ages', 'Industry'])


def test_name_labels_and_footnotes():
    table = Table('example', ['Group > Score'], [Row('Region > MIỀN NAM (1)*', [1])])
    labels = table_labels([table])
    assert 'Score' in labels and 'MIỀN NAM (1)*' in labels
    assert not check_answer('miền nam†', 'Which region?', labels)
    assert not check_answer('SCORE[2]', 'What is the name of the column?', labels)
    issues = check_answer('MIERNAM', 'Which region?', labels)
    assert 'Name/label' in issues[0]['reason']
    assert 'MIỀN NAM (1)*' in issues[0]['closest_labels']
    assert not check_answer('42', 'Which number is largest?', labels)
    assert not check_answer('Unknown', 'Describe the table.', labels)


@pytest.mark.parametrize('code', [
    "answer = df.index.str.contains('MIERNAM')",
    "answer = df.index.str.startswith('MIERNAM')",
    "answer = df.index == 'MIERNAM'",
    "answer = 'MIERNAM' == df.index",
    "```python\nanswer = df.index == 'MIERNAM'\n```",
])
def test_unknown_program_literals(code):
    assert missing_program_labels(code, LABELS) == ['MIERNAM']
    issues = check_answer('12', 'Find a value.', LABELS, code)
    assert 'Program label predicate' in issues[0]['reason']
    assert 'MIỀN NAM' in issues[0]['closest_labels']


def test_normalization_and_valid_predicates():
    decomposed = unicodedata.normalize('NFD', 'miền nam')
    assert normalize_label(decomposed) == normalize_label('MIỀN NAM')
    assert not missing_program_labels(f'answer = df.index == {decomposed!r}', LABELS)
    assert not missing_program_labels("answer = df.index.str.contains('NAM')", LABELS)
    assert not missing_program_labels("answer = df.index.str.startswith('Region')", LABELS)
    assert not missing_program_labels("answer = df.index == label", LABELS)
    assert not missing_program_labels('answer = df.index == 0', LABELS)
    assert not missing_program_labels('invalid python!', LABELS)
    assert closest_labels('MIERNAM', LABELS)[0] == 'MIỀN NAM'


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.budgets = []

    def complete(self, messages, max_cost=None, **kwargs):
        self.messages.append([dict(message) for message in messages])
        self.budgets.append(max_cost)
        return next(self.responses)


def run_sample(monkeypatch, responses, enabled=True, **overrides):
    config = load_config()
    config.update(sanity_retry=enabled, max_attempts=1, **overrides)
    client = StubClient(responses)
    events = []
    monkeypatch.setattr('solver.strategies.run_program',
                        lambda code, *args, **kwargs: {'answer': int(code.split('=')[1])})
    result = program(client=client, messages=[{'role': 'user', 'content': 'table'}],
                     query={'id': 'q', 'query': 'How many workers?'}, path='unused.xlsx',
                     config=config, budget=.03, stopped=threading.Event(),
                     on_event=lambda kind, data: events.append((kind, data)),
                     tables=[Table('test', ['Workers'], [Row('MIỀN NAM', [12])])])
    return result, client, events


@pytest.mark.parametrize('retry,expected,accepted', [('12', '12', True), ('0', '0', False)])
def test_retry_zero_and_keep_suspicious_original(monkeypatch, retry, expected, accepted):
    result, client, events = run_sample(monkeypatch, [('answer = 0', .001),
                                                     (f'answer = {retry}', .002)])
    answer, cost, attempts, error = result
    assert answer == expected and error is None
    assert cost == pytest.approx(.003)
    assert client.budgets == pytest.approx([.03, .029])
    feedback = client.messages[1][-1]['content']
    assert 'exactly zero' in feedback and 'Closest existing labels' in feedback
    assert 'Question your assumption: re-derive the row/column selection with exact labels from the table, possibly with a different approach.' in feedback
    assert attempts[0]['sanity']
    assert attempts[1]['sanity_retry'] and attempts[1]['sanity_accepted'] == accepted
    assert len([event for event in events if event[0] == 'sanity']) == 2


@pytest.mark.parametrize('enabled,overrides', [(False, {}), (True, {'sanity_max_retries': 0})])
def test_no_extra_call_when_disabled(monkeypatch, enabled, overrides):
    result, client, _ = run_sample(monkeypatch, [('answer = 0', .001)], enabled, **overrides)
    assert result[0] == '0' and len(client.messages) == 1


def test_no_retry_when_budget_spent(monkeypatch):
    result, client, _ = run_sample(monkeypatch, [('answer = 0', .03)])
    assert result[0] == '0' and len(client.messages) == 1


def test_failed_retry_keeps_original(monkeypatch):
    result, client, _ = run_sample(monkeypatch, [('answer = 0', .001), ('bad code', .001)])
    assert result[0] == '0' and result[3] is None and len(client.messages) == 2
    assert result[2][-1]['error']


def test_no_retry_for_clean_candidate(monkeypatch):
    result, client, _ = run_sample(monkeypatch, [('answer = 12', .001)])
    assert result[0] == '12' and len(client.messages) == 1


def test_deadline_prevents_retry(monkeypatch):
    from solver.config import DeadlineStop
    import time
    stop = DeadlineStop(threading.Event(), time.monotonic() + 60)
    def execute(*args, **kwargs):
        stop.deadline = time.monotonic() - 1
        return {'answer': 0}
    monkeypatch.setattr('solver.strategies.run_program', execute)
    config = load_config()
    config.update(sanity_retry=True)
    client = StubClient([('answer = 0', .001)])
    result = program(client=client, messages=[], query={'id': 'q', 'query': 'How many?'},
                     path='unused', config=config, budget=.03, stopped=stop,
                     on_event=lambda *args: None, tables=[])
    assert result[0] == '0' and len(client.messages) == 1


def test_suspicious_retry_retains_different_original(monkeypatch):
    config = load_config()
    config.update(sanity_retry=True, max_attempts=1)
    client = StubClient([("answer = df.index.str.contains('MIERNAM')", .001),
                         ('answer = 0', .001)])
    monkeypatch.setattr('solver.strategies.run_program',
                        lambda code, *args, **kwargs: {'answer': 7 if 'MIERNAM' in code else 0})
    result = program(client=client, messages=[], query={'id': 'q', 'query': 'How many?'},
                     path='unused', config=config, budget=.03, stopped=threading.Event(),
                     on_event=lambda *args: None,
                     tables=[Table('test', ['Workers'], [Row('MIỀN NAM', [7])])])
    assert result[0] == '7'
    assert 'MIERNAM' in client.messages[1][-1]['content']
    assert 'MIỀN NAM' in client.messages[1][-1]['content']
    assert not result[2][-1]['sanity_accepted']


def test_pipeline_logs_sanity(data_dir, tmp_path):
    from solver.pipeline import Pipeline
    config = load_config()
    config.update(strategy='program', votes=1, sanity_retry=True, max_attempts=1)
    client = StubClient([('answer = 0', .001), ('answer = len(tables[0])', .001)])
    pipeline = Pipeline(client, config=config, log_path=tmp_path/'questions.jsonl')
    row = pipeline.solve({'id': 'sanity_q', 'file_name': 'p4_t01.xlsx',
                          'query': 'How many workers? Answer with only the integer.'}, data_dir)
    assert row['answer'] == '20'
    attempts = json.loads(pipeline.log_path.read_text())['attempts']
    assert attempts[0]['sanity'] and attempts[1]['sanity_retry']
    assert any(json.loads(line)['kind'] == 'sanity' for line in pipeline.trace_path.read_text().splitlines())


@pytest.mark.parametrize('override', ['sanity_retry=1', 'sanity_max_retries=-1',
                                     'sanity_max_retries=true', 'sanity_max_retries=1.5'])
def test_config_validation(override):
    with pytest.raises(ValueError):
        apply_overrides(load_config(), [override])
