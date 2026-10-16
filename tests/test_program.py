import ast
import csv
import json
from pathlib import Path
import pytest
from solver.program import run_program, build_prompt, validate_code
from solver.parser import parse_workbook
from solver.formatter import format_answer


@pytest.mark.parametrize('n', range(1,21))
def test_oracle(n, data_dir, oracle_programs):
    queries=json.loads((data_dir/'queries.json').read_text())
    q=queries[n-1]
    with (data_dir/'sample_submission.csv').open(encoding='utf-8-sig',newline='') as f:
        gold={r['id']:r['answer'] for r in csv.DictReader(f)}
    code=(oracle_programs/f"{q['id']}.py").read_text()
    assert not any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(ast.parse(code)))
    result=run_program(code, data_dir/'xlsx_files'/q['file_name'])
    assert 'error' not in result, result
    assert format_answer(result['answer'],q['query']) == gold[q['id']]


def test_executor_failure_and_timeout(data_dir):
    p=data_dir/'xlsx_files/p4_t01.xlsx'
    assert 'error' in run_program('answer = 1/0',p)
    assert 'error' in run_program('while True: pass',p, timeout=1)
    assert run_program('```python\nanswer = len(tables[0])\n```',p) == {'answer':20}


def test_executor_solve_without_answer(data_dir):
    p=data_dir/'xlsx_files/p4_t01.xlsx'
    assert run_program('def solve(tables):\n    return len(tables[0])',p) == {'answer':20}


@pytest.mark.parametrize('code,expected,stderr', [
    ('answer = 42\nprint(answer)', {'answer': 42}, '42\n'),
    ('print(42)', {'answer': None}, '42\n'),
    ('def solve():\n    print("debug", 42, sep=":", end="!", flush=True)\n    return 42',
     {'answer': 42}, 'debug:42!'),
    ('answer = 42\nprint(answer, file=None)', {'answer': 42}, '42\n'),
])
def test_executor_print_uses_stderr(data_dir, monkeypatch, code, expected, stderr):
    import subprocess
    real_run = subprocess.run
    children = []
    def capture(*args, **kwargs):
        child = real_run(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr('solver.program.subprocess.run', capture)
    assert run_program(code, data_dir/'xlsx_files/p4_t01.xlsx') == expected
    assert children[0].stderr == stderr
    assert json.loads(children[0].stdout) == expected


@pytest.mark.parametrize('code', ['import os', "answer = open('/etc/passwd').read()", 'answer = tables[0].to_csv()', 'answer = pd.read_excel("x")', 'answer = pd.__dict__', 'from pandas import read_csv', 'answer = pd.ExcelFile("x")'])
def test_restrictions(code):
    with pytest.raises(ValueError): validate_code(code)


def test_prompt_size(data_dir):
    for q in json.loads((data_dir/'queries.json').read_text()):
        messages=build_prompt(q['query'],parse_workbook(data_dir/'xlsx_files'/q['file_name']),q['id'])
        assert sum(len(m['content']) for m in messages) < 15500
        assert 'Columns:' in messages[1]['content']


def test_oracle_uses_changed_parsed_data(data_dir,tmp_path,oracle_programs):
    # Mutate a value in a copied workbook: the oracle must compute a new answer.
    import openpyxl
    src=data_dir/'xlsx_files/p4_t08.xlsx'
    w=openpyxl.load_workbook(src); w.active['C8']=321
    p=tmp_path/'changed.xlsx'; w.save(p)
    code=(oracle_programs/'p4_q015.py').read_text()
    assert run_program(code,p) == {'answer':321}


def test_safe_pandas_conversions(data_dir):
    p=data_dir/'xlsx_files/p4_t01.xlsx'
    result=run_program("answer = pd.to_datetime(tables[0]['Date of > Birth']).dt.year.min()",p)
    assert result == {'answer':1995}


def test_label_helpers_and_candidates(data_dir):
    from solver.program import find_row, find_col
    import pandas as pd
    df=pd.DataFrame([[1,2],[3,4]], index=['Men > Rate(1)', 'Women > Rate'], columns=['2024 > Value', '2023 > Value'])
    assert find_row(df,'RATE',section='Women') == 'Women > Rate'
    assert find_row(df,'Men > Rate') == 'Men > Rate(1)'
    assert find_col(df,'2024','value') == '2024 > Value'
    for lookup in (lambda: find_row(df,'rate'), lambda: find_col(df,'missing')):
        with pytest.raises(ValueError,match='Candidates:') as exc: lookup()
        assert '2024 > Value' in str(exc.value) or 'Women > Rate' in str(exc.value)
    result=run_program("df=tables[0]\nanswer=df.loc[find_row(df, '5 to 14 hours'), find_col(df, 'Persons at work', 'Agriculture')]",data_dir/'xlsx_files/p4_t08.xlsx')
    assert result == {'answer':134}


def test_prompt_exact_labels(data_dir):
    table=parse_workbook(data_dir/'xlsx_files/p4_t01.xlsx')[0]
    text=build_prompt('Example', [table])[1]['content']
    assert 'c0' not in text
    assert repr(table.header_paths) in text
    assert "0 | '1' |" in text
