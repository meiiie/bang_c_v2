import importlib.util
import json
from pathlib import Path
import threading

import pytest

from solver.config import load_config, apply_overrides
from solver.fast_path import lookup
from solver.parser import Table, Row, parse_workbook
from solver.pipeline import Pipeline
from solver.program import build_prompt
from solver.strategies import vote, verified_program


class Client:
    def __init__(self, responses, barrier=None):
        self.responses = iter(responses)
        self.calls = []
        self.lock = threading.Lock()
        self.barrier = barrier

    def complete(self, messages, max_cost=None, temperature=0):
        with self.lock:
            index = len(self.calls)
            self.calls.append((messages, max_cost, temperature))
            response = next(self.responses)
        if self.barrier and index < 2:
            self.barrier.wait(timeout=3)
        return response


@pytest.fixture
def context(monkeypatch):
    # Real execution is covered below; these tests isolate orchestration.
    def execute(code, path, **kwargs):
        code = code.replace('```python', '').replace('```', '').strip()
        if code == 'bad': return {'error': 'bad code'}
        return {'answer': json.loads(code.split('=', 1)[1])}
    monkeypatch.setattr('solver.strategies.run_program', execute)
    config = load_config()
    config.update(votes=3, max_votes=3, max_attempts=1, vote_mode='adaptive', vote_first=2)
    return dict(messages=[{'role':'system','content':'Compute using Python'},
                          {'role':'user','content':'Query ID: q\nQuestion: integer\nColumns: ["Value"]\nX | 1'}],
                query={'id':'q','query':'Answer with only the integer, without commas or units.'},
                path=Path('unused.xlsx'), config=config, budget=.03,
                stopped=threading.Event(), on_event=lambda *args: None)


def test_adaptive_parallel_agreement_and_log(context):
    client = Client([('answer=1',.001),('answer=1.0',.001)], threading.Barrier(2))
    events=[]
    context.update(client=client,on_event=lambda k,d:events.append((k,d)))
    answer,cost,attempts,error=vote(**context)
    assert (answer,error)==('1',None)
    assert cost==pytest.approx(.002) and len(client.calls)==2
    assert events[-1][1]['samples_used']==2
    assert {a['sample'] for a in attempts}=={0,1}


@pytest.mark.parametrize('mode,max_votes,responses,expected,calls',[
    ('fixed',3,[1,1,2],'1',3),
    ('adaptive',3,[1,2,2],'2',3),
    ('adaptive',2,[1,2],'1',2),
    ('adaptive',4,[1,2,2,2],'2',4),
])
def test_vote_modes_and_cap(context,mode,max_votes,responses,expected,calls):
    context['config'].update(vote_mode=mode,votes=len(responses),max_votes=max_votes)
    client=Client([(f'answer={v}',.001) for v in responses])
    assert vote(**dict(context,client=client))[0]==expected
    assert len(client.calls)==calls


def test_failed_sample_cannot_early_stop(context):
    client=Client([('bad',.001),('bad',.001),('answer=7',.001)])
    assert vote(**dict(context,client=client))[0]=='7'
    assert len(client.calls)==3


@pytest.mark.parametrize('direct,expected',[('2','2'),('4','4'),('','1'),('```python\nanswer=2\n```','1')])
def test_hybrid_only_on_no_majority(context,direct,expected):
    context['config']['hybrid_text']=True
    client=Client([('answer=1',.001),('answer=2',.001),('answer=3',.001),(direct,.001)])
    result=vote(**dict(context,client=client))
    assert result[0]==expected and len(client.calls)==4
    assert 'answer only' in client.calls[-1][0][0]['content']
    assert 'Columns:' in client.calls[-1][0][1]['content']
    assert result[2][-1]['kind']=='hybrid_text'


def test_hybrid_skips_majority(context):
    context['config']['hybrid_text']=True
    client=Client([('answer=1',.001),('answer=2',.001),('answer=2',.001)])
    assert vote(**dict(context,client=client))[0]=='2'
    assert len(client.calls)==3


@pytest.mark.parametrize('correction,expected',[('OK','1'),('```python\nanswer=2\n```','2'),
                                              ('```python\nanswer=4\n```','1'),('bad','1')])
def test_verifier_needs_sample_support(context,correction,expected):
    context['config']['verifier']=True
    client=Client([('answer=1',.001),('answer=2',.001),('answer=3',.001),(correction,.001)])
    answer,cost,attempts,_=vote(**dict(context,client=client))
    assert answer==expected and len(client.calls)==4
    assert attempts[-1]['kind']=='verifier' and cost==pytest.approx(.004)
    assert 'Proposed code:' in client.calls[-1][0][-1]['content']
    assert 'Formatted answer: 1' in client.calls[-1][0][-1]['content']


def test_verifier_single_sample_prefers_original(context):
    context['config']['verifier']=True
    client=Client([('answer=1',.001),('```python\nanswer=2\n```',.001)])
    assert verified_program(**dict(context,client=client))[0]=='1'
    assert len(client.calls)==2


def test_auxiliary_budget_and_deadline(context):
    context['config'].update(hybrid_text=True,verifier=True)
    client=Client([('answer=1',.006),('answer=2',.006),('answer=3',.006),('2',.006),('OK',.006)])
    result=vote(**dict(context,client=client))
    assert result[1]==pytest.approx(.03)
    assert all(call[1]<=.006 for call in client.calls)
    context['stopped'].set()
    client=Client([])
    assert vote(**dict(context,client=client))[0] is None
    assert client.calls==[]


def test_verifier_real_sandbox(data_dir,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    config=load_config(); config.update(votes=3,verifier=True,max_attempts=1,vote_first=2)
    client=Client([('answer=1',0),('answer=2',0),('answer=3',0),('```python\nanswer=2\n```',0)])
    p=Pipeline(client,config=config)
    assert p.solve({'id':'q','file_name':'p4_t01.xlsx','query':'Answer with only the integer.'},data_dir)['answer']=='2'
    record=json.loads(p.log_path.read_text())
    assert record['samples_used']==3 and record['attempts'][-1]['result']=={'answer':2}


def test_compact_schema_prefix_empty_values_and_number_format():
    table=Table('Test:1',['2023 > Jobs','2023 > Rate','2023 > Empty'],
                [Row('Men > A',[12.0,None,None]),Row('Women > A',[4.0,2.5,None])])
    text=table.to_text(serialization='compact')
    assert 'Columns: '+repr(table.header_paths) in text
    assert "Table column prefix: '2023 > '" in text and 'Header ids: [0, 1, 2]' in text
    assert 'Men > A | 12\n' in text and 'Women > A | 4 | 2.5' in text
    assert '12.0' not in text
    assert table.to_dataframe().columns.tolist()==table.header_paths
    assert len(table.to_text(100,serialization='compact'))<=100
    config=load_config(); config['serialization']='compact'
    assert 'Table column prefix:' in build_prompt('Jobs?', [table],config=config)[1]['content']
    config['serialization']='pipe'
    assert 'Column positions:' in build_prompt('Jobs?', [table],config=config)[1]['content']


@pytest.fixture
def table():
    return Table('Test:1',['2023 > Units','2023 > Price'],[Row('North > Maple',[12,3]),Row('South > Birch',[4,5])])


@pytest.mark.parametrize('question',[
    'What is the number of "Units" in "Maple"?',
    'What is Units for Maple?',
    'What is the value of "2023 > Units" for "North > Maple"? Answer with only the integer, without commas or units.',
])
def test_strict_fast_path(table,question):
    assert lookup(question,[table])=='12'


@pytest.mark.parametrize('question',[
    'What is the sum of Units in Maple?',
    'What is Units for Maple and Birch?',
    'What is Units for Maple? Divide by Price.',
    'What is the percentage of Units for Maple?',
    'Which row has the highest Units?',
    'What is Units for Maple as of 2023?',
])
def test_fast_path_rejects_computations(table,question):
    assert lookup(question,[table]) is None


def test_fast_path_global_ambiguity(table):
    question='What is Units for Maple?'
    assert lookup(question,[table,table]) is None
    table.rows.append(Row('South > Maple',[8,4]))
    assert lookup(question,[table]) is None
    assert lookup('What is Units for "North > Maple"?',[table])=='12'
    table.header_paths.append('2024 > Units')
    assert lookup('What is Units for "North > Maple"?',[table]) is None


def test_fast_path_pipeline_skips_client(table,tmp_path,monkeypatch,data_dir):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('solver.pipeline.parse_workbook',lambda *args:[table])
    config=load_config(); config.update(fast_path=True,verifier=True,prompt_char_budget=1)
    client=Client([]); pipeline=Pipeline(client,config=config)
    q={'id':'q','file_name':'p4_t01.xlsx','query':'What is Units for Maple?'}
    assert pipeline.solve(q,data_dir)['answer']=='12' and not client.calls
    assert json.loads(pipeline.log_path.read_text())['samples_used']==0


@pytest.mark.parametrize('override',['vote_mode=unknown','serialization=unknown','max_votes=0','hybrid_text=1','verifier="yes"','fast_path=null'])
def test_invalid_flags(override):
    with pytest.raises(ValueError):apply_overrides(load_config(),[override])


def test_hard_pack_gold_and_layout(data_dir,tmp_path,capsys):
    builder = Path(__file__).resolve().parents[1]/'tools/make_robust_set.py'
    if not builder.is_file():
        pytest.skip('Optional robust-set generator not installed')
    spec=importlib.util.spec_from_file_location('make_robust_set', builder)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    b=module.Builder(data_dir,tmp_path)
    b.hard(); b.finish()
    queries=json.loads((tmp_path/'queries.json').read_text())
    assert len(queries)>=40 and queries[0]['id']=='h_q001'
    assert len({q['query'] for q in queries})==len(queries)
    assert all('sheet' not in q['query'].lower() and not __import__('re').search(r'\b[A-Z]\d+\b',q['query']) for q in queries)
    gold={r['id']:r['answer'] for r in b.gold}
    assert gold['h_q013']=='15570'  # Parent differs from rounded parts (15571).
    assert gold['h_q045']=='205' and gold['h_q048']=='375' and gold['h_q051']=='240'
    tables=parse_workbook(tmp_path/'xlsx_files/h_people_orders.xlsx')
    assert len(tables)==2
    assert tables[0].rows[0].label_path=='An' and tables[1].rows[0].label_path=='O11'
    assert tables[0].rows[-1].label_path=='Total'
    assert gold['h_q055']=='Binh' and gold['h_q056']=='Lan'
    assert all(e['source'] for e in b.evidence)
