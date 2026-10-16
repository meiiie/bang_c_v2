"""Regressions against source cells, independent of solver answers."""
from datetime import datetime
from pathlib import Path

import openpyxl
import pytest

from solver.config import load_config
from solver.formatter import format_answer
from solver.helpers import sheet
from solver.parser import parse_workbook, is_data
from solver.program import build_prompt, run_program

ROOT = Path(__file__).resolve().parents[2] / 'adversarial' / 'xlsx_files'


def tables(name):
    if not ROOT.is_dir():
        pytest.skip('External adversarial workbooks not installed')
    return parse_workbook(ROOT / (name + '.xlsx'))


def test_regional_numeric_text_and_exact_sheet_names():
    ts = tables('regional_sales')
    assert len(ts) == 3
    df = sheet([t.to_dataframe() for t in ts], 'Sales 2025')
    assert df.loc['MIỀN NAM > Huế', '2025 revenue'] == 5542
    assert df.loc['MIỀN BẮC > Hà Nội', 'Margin (%)'] == 12.5
    prompt = build_prompt('Revenue in Sales 2025?', ts)[1]['content']
    for name in ['Sales 2025', 'Sales 2025 draft', 'Sales 2024']:
        assert f'Sheet: {name};' in prompt
    assert is_data(['Branch', '5,542', '12.5%', 18, 'Reviewed'])
    with pytest.raises(ValueError):
        sheet([t.to_dataframe() for t in ts], 'Sales')


@pytest.mark.parametrize('name,column,coordinates', [
    ('signed_accounts', 'Adjustment', ['C6','C8','C13','C15','C20','C22']),
    ('stacked_operations', 'Variance million VND', ['D13','D15']),
])
def test_accounting_values_survive_all_sheets(name, column, coordinates):
    ts = tables(name)
    source = openpyxl.load_workbook(ROOT / (name + '.xlsx'))
    try:
        for ws in source:
            t = next(t for t in ts if t.metadata['sheet_name'] == ws.title and column in t.header_paths)
            values = [r.values[t.header_paths.index(column)] for r in t.rows]
            for coord in coordinates:
                assert -int(ws[coord].value.strip('()').replace(',', '')) in values
    finally:
        source.close()


def test_mixed_year_merged_headers():
    for t in tables('merged_programs'):
        assert len(t.header_paths) == len(set(t.header_paths)) == 8
        assert t.header_paths[:2] == ['Actual > Participants > 2024', 'Actual > Participants > 2025']
        assert len(t.rows) == 15
        assert not any(r.label_path == 'Program' for r in t.rows)


@pytest.mark.parametrize('name', ['cpsaat33', 'cpsaat37'])
def test_race_sex_hierarchy_and_unique_keys(name):
    t = tables(name)[0]
    df = t.to_dataframe()
    assert df.index.is_unique
    for race in ['White', 'Black or African American', 'Asian', 'Hispanic or Latino ethnicity']:
        for sex in ['Men', 'Women']:
            assert f'RACE AND HISPANIC OR LATINO ETHNICITY > {race} > {sex}' in df.index


@pytest.mark.parametrize('name,path', [
    ('cpsaat09', 'Management, professional, and related occupations > Management, business, and financial operations occupations > Management occupations'),
    ('cpsaat12', 'CLASS OF WORKER > Nonagricultural industries > Wage and salary workers > Private industries > Private households'),
])
def test_cell_indentation(name, path):
    assert path in tables(name)[0].to_dataframe().index


def test_date_contract_and_helper():
    from solver.helpers import helper_namespace
    ts = tables('personnel_dates')
    for t in ts:
        assert 'All text dates use dd/mm/yyyy.' in t.to_text()
        assert t.rows[0].values[1] == '04/05/2021'
    helper = helper_namespace()['parse_date']
    assert helper('04/05/2021') == datetime(2021, 5, 4)
    assert helper('04/05/2021', dayfirst=False) == datetime(2021, 4, 5)
    assert run_program("answer=parse_date(tables[0].iloc[0, 1])", ROOT/'personnel_dates.xlsx') == {'answer': '2021-05-04T00:00:00'}


@pytest.mark.parametrize('raw,query,expected', [
    (689.125, '2 decimals', '689.13'), ('689.125', 'two decimal places', '689.13'),
    (2.5, 'integer', '3'), (-2.5, 'integer', '-3'), ('2.5', 'integer', '3'),
    ('(649)', 'integer', '-649'), ('−2.5', 'integer', '-3'),
])
def test_half_up(raw, query, expected):
    assert format_answer(raw, query) == expected


def test_normalization_placeholders_and_label_only_footnotes(tmp_path):
    w = openpyxl.Workbook(); s = w.active
    s.append(['Label', 'Value', 'Text'])
    inputs = ['5,542', '12.5%', '(1,234)', '(649)', '-', '–', '—', 'NA', 'n.a.', '−42', '04/05/2021']
    expected = [5542, 12.5, -1234, -649, None, None, None, None, None, -42, '04/05/2021']
    for i, value in enumerate(inputs):
        s.append([f'Label {i}(2)', value, 'note(2)'])
    p = tmp_path/'values.xlsx'; w.save(p)
    t = parse_workbook(p)[0]
    assert [r.values[0] for r in t.rows] == expected
    assert all('(2)' not in r.label_path and r.values[1] == 'note(2)' for r in t.rows)
    assert t.header_paths[0] == 'Value'  # Mixed numeric/percent column has no inferred unit.


@pytest.mark.parametrize('rows', [
    [['Label','Text'], ['alpha','hello'], ['beta','world']],
    [['Title'], ['alpha'], ['beta']],
    [[None,'Title'], [None,'hello']],
])
def test_raw_grid_fallback(tmp_path, rows):
    w = openpyxl.Workbook(); s = w.active
    for row in rows: s.append(row)
    p = tmp_path/'raw.xlsx'; w.save(p)
    ts = parse_workbook(p)
    assert len(ts) == 1
    text = build_prompt('Show sheet content', ts)[1]['content']
    for row in rows:
        for value in row:
            if value: assert value in text


def test_spaces_sections_and_duplicate_label_cleanup(tmp_path):
    w = openpyxl.Workbook(); s = w.active
    for row in [['Label','Value'], ['SECTION',None], ['Parent',10], ['  Child(1)',2], ['  Child(2)',3], ['OTHER',None], ['Parent',20], ['  Child',4], ['  Child #2',5], ['  Child',6]]:
        s.append(row)
    p = tmp_path/'indent.xlsx'; w.save(p)
    df = parse_workbook(p)[0].to_dataframe()
    assert df.index.is_unique
    assert 'SECTION > Parent > Child' in df.index
    assert 'SECTION > Parent > Child #2' in df.index
    assert 'OTHER > Parent > Child' in df.index


def test_normalize_stage_order():
    from solver.parser import STAGES
    names = [fn.__name__ for fn in STAGES]
    assert 'normalize_values' in names
    assert names.index('normalize_values') < names.index('detect_headers')
    assert load_config()['stages'] == names


@pytest.mark.parametrize('raw,query,expected', [
    ('1.25e2', 'two decimal places', '125.00'),
    ('(649)', 'number', '-649'),
    (True, 'integer', '1'),
])
def test_numeric_answer_forms(raw, query, expected):
    assert format_answer(raw, query) == expected


def test_all_adversarial_sheets_present_with_unique_index():
    if not ROOT.is_dir():
        pytest.skip('External adversarial workbooks not installed')
    for path in ROOT.glob('*.xlsx'):
        source = openpyxl.load_workbook(path, data_only=True)
        try:
            expected = {ws.title for ws in source if any(c.value not in (None, '') for row in ws for c in row)}
        finally:
            source.close()
        parsed = parse_workbook(path)
        assert {t.metadata['sheet_name'] for t in parsed} == expected, path.name
        assert all(t.to_dataframe().index.is_unique for t in parsed), path.name
