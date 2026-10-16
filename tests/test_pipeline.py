import csv
import json
import shutil
from solver.config import initial_rows, load_config, normalize_query_ids, write_csv
from solver.llm import MockClient, NoLLM
from solver.pipeline import Pipeline
import pytest


class StubClient:
    def __init__(self, responses):
        self.responses=iter(responses)
        self.messages=[]
        self.budgets=[]

    def complete(self, messages, max_cost=None, temperature=0, **kwargs):
        self.messages.append([dict(m) for m in messages])
        self.budgets.append(max_cost)
        return next(self.responses)


def question():
    return {'id':'test_q', 'file_name':'p4_t01.xlsx', 'query':'Answer with only the integer score.'}


def test_retry_with_feedback(data_dir, tmp_path):
    c=StubClient([('answer = 1/0',.001), ('answer = tables[0]["Score"].max()',.001)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl')
    assert p.solve(question(), data_dir)['answer'] == '99'
    assert 'ZeroDivisionError' in c.messages[1][-1]['content']
    assert c.budgets == pytest.approx([.03,.029])
    log=json.loads(p.log_path.read_text())
    assert len(log['attempts']) == 2 and log['cost'] == .002


def test_failures_always_write_answer(data_dir,tmp_path):
    p=Pipeline(MockClient(),log_path=tmp_path/'log.jsonl',config=dict(load_config(),rescue_pass=False))
    result=p.solve_all([question(),dict(question(),id='missing',file_name='absent.xlsx')],data_dir,workers=2)
    assert [r['answer'] for r in result] == ['0','0']
    assert len(p.log_path.read_text().splitlines()) == 2


def test_budget_stops(data_dir,tmp_path):
    c=StubClient([('answer = 1/0',.03)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl')
    assert p.solve(question(),data_dir)['answer'] == '0'
    assert len(c.messages) == 1
    with pytest.raises(ValueError): Pipeline(c,budget=.11)


def test_self_consistency(data_dir,tmp_path):
    c=StubClient([('answer = len(tables[0])',.001),('answer = tables[0].shape[0]',.001)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl',votes=2,config=dict(load_config(),vote_first=2))
    assert p.solve(question(),data_dir)['answer'] == '20'
    assert len(c.messages) == 2


def test_dump_all(data_dir,tmp_path):
    queries=json.loads((data_dir/'queries.json').read_text())
    p=Pipeline(MockClient(),log_path=tmp_path/'log.jsonl')
    rows=p.solve_all(queries,data_dir,mode='dump',dump_dir=tmp_path/'prompts')
    files=list((tmp_path/'prompts').glob('*.txt'))
    assert len(files) == 20 and all(r['answer'] == '' for r in rows)
    for f in files:
        assert len(f.read_text()) < 9200


def test_normalize_query_ids():
    queries=[dict(question(),id=qid) for qid in (1, 0, 'p4_q001', '', None)]
    normalized=normalize_query_ids(queries)
    assert [q['id'] for q in normalized] == ['1','0','p4_q001','','']
    assert [q['id'] for q in queries] == [1,0,'p4_q001','',None]
    assert all(q['file_name'] == question()['file_name'] and q['query'] == question()['query'] for q in normalized)


def test_integer_query_ids(data_dir,tmp_path):
    dataset=tmp_path/'data'
    (dataset/'xlsx_files').mkdir(parents=True)
    shutil.copy2(data_dir/'xlsx_files'/question()['file_name'],dataset/'xlsx_files'/question()['file_name'])
    original=[dict(question(),id=qid) for qid in (1,2)]
    (dataset/'queries.json').write_text(json.dumps(original))
    queries=json.loads((dataset/'queries.json').read_text())
    expected=[str(q['id']) for q in original]
    p=Pipeline(MockClient(),log_path=tmp_path/'log.jsonl')
    rows=p.solve_all(queries,dataset,mode='dump',dump_dir=tmp_path/'prompts')
    assert [row['id'] for row in rows] == expected
    assert all((tmp_path/'prompts'/f'{qid}.txt').is_file() for qid in expected)
    assert all(json.loads(line)['error'] is None for line in p.log_path.read_text().splitlines())
    fallback=initial_rows(queries)
    assert [row['id'] for row in fallback] == expected
    write_csv(tmp_path/'answers.csv',fallback)
    with (tmp_path/'answers.csv').open(newline='') as stream:
        assert [row['id'] for row in csv.DictReader(stream)] == expected


@pytest.mark.parametrize('qid', [1, 'p4_q001', '', None, 'bad/id'])
def test_single_query_id(data_dir,tmp_path,qid):
    p=Pipeline(MockClient(),log_path=tmp_path/'log.jsonl')
    row=p.solve(dict(question(),id=qid),data_dir,mode='dump',dump_dir=tmp_path/'prompts')
    assert row['id'] == ('' if qid is None else str(qid))
    record=json.loads(p.log_path.read_text())
    assert record['error'] == ('ValueError: Invalid query ID' if qid in ('', None, 'bad/id') else None)


def test_mock_file(tmp_path):
    path=tmp_path/'programs.json'; path.write_text('{"test_q": "answer = len(tables[0])"}')
    c=MockClient(path)
    assert c.complete([{'content':'Query ID: test_q'}]) == ('answer = len(tables[0])',0)
    with pytest.raises(NoLLM): MockClient().complete([])


def test_consistency_majority(data_dir,tmp_path, monkeypatch):
    monkeypatch.setenv("SOLVER_VOTES", "3")
    c=StubClient([('answer = len(tables[0])',.001),('answer = len(tables[0]) + 1',.001),('answer = tables[0].shape[0]',.001)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl',self_consistency=True,config=dict(load_config(),vote_first=2))
    assert p.solve(question(),data_dir)['answer'] == '20'
    assert len(c.messages) == 3


def test_empty_length_response_is_logged_and_retried(data_dir,tmp_path):
    usage={'prompt_tokens':100, 'completion_tokens':4000, 'reasoning_tokens':4000, 'finish_reason':'length'}
    c=StubClient([('', .001, usage), ('answer = True', .001, {})])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl')
    assert p.solve(dict(question(),query='Answer with only Yes or No.'),data_dir)['answer'] == 'Yes'
    record=json.loads(p.log_path.read_text())
    assert record['attempts'][0]['usage'] == usage
    assert 'finish_reason=length' in c.messages[1][-1]['content']
    assert 'Empty answer' in record['attempts'][0]['error']


def test_helper_candidates_in_retry_feedback(data_dir,tmp_path):
    c=StubClient([("answer = find_col(tables[0], 'missing')", .001), ('answer = 1', .001)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl')
    assert p.solve(question(),data_dir)['answer'] == '1'
    assert 'Candidates:' in c.messages[1][-1]['content']
    assert 'Score' in c.messages[1][-1]['content']


def test_votes_parallel_independent_and_budgeted(monkeypatch,data_dir,tmp_path):
    import threading
    barrier=threading.Barrier(3)
    class ParallelClient:
        def complete(self,messages,max_cost=None,temperature=0):
            assert max_cost == pytest.approx(.01)
            assert temperature > 0
            assert len(messages) == 2
            barrier.wait(timeout=5)
            return 'answer = 1', .001, {}
    monkeypatch.setenv('SOLVER_VOTES','3')
    monkeypatch.setenv('SOLVER_VOTE_MODE','fixed')
    p=Pipeline(ParallelClient(),log_path=tmp_path/'log.jsonl')
    assert p.solve(question(),data_dir)['answer'] == '1'
    record=json.loads(p.log_path.read_text())
    assert len(record['attempts']) == 3
    assert record['cost'] == pytest.approx(.003)


def test_direct_keyerror_has_candidate_feedback(data_dir,tmp_path):
    c=StubClient([('answer = tables[0]["row label"]', .001), ('answer = 1', .001)])
    p=Pipeline(c,log_path=tmp_path/'log.jsonl')
    assert p.solve(question(),data_dir)['answer'] == '1'
    assert 'KeyError' in c.messages[1][-1]['content']
    assert 'Candidates:' in c.messages[1][-1]['content']
    assert 'Score' in c.messages[1][-1]['content']


import pytest as _pytest

@_pytest.fixture(autouse=True)
def _single_vote(monkeypatch):
    monkeypatch.setenv('SOLVER_VOTES', '1')


def test_aggregate_label_guard():
    from solver.strategies import _aggregate_label
    assert _aggregate_label('Total', 'Which product had the highest margin? Answer with only the name.')
    assert not _aggregate_label('Cocoa', 'Which product had the highest margin?')
    assert not _aggregate_label('Total, 16 years and over', 'What is the number of workers in "Total, 16 years and over"?')
