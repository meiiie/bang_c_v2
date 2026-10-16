"""Small synthetic regressions for totals, footnotes and fragment headers."""
import openpyxl
import pytest

from solver.parser import clean_text, parse_workbook
from solver.program import build_prompt


def workbook(tmp_path, rows):
    book = openpyxl.Workbook()
    for row in rows:
        book.active.append(row)
    path = tmp_path / 'layout.xlsx'
    book.save(path)
    book.close()
    return parse_workbook(path)[0]


@pytest.mark.parametrize('raw,expected', [
    ('Example sector7  ', 'Example sector'),
    ('Resource category12,4 ', 'Resource category'),
    ('Public administration3', 'Public administration'),
    ('Frequency rates1', 'Frequency rates'),
    ('Classification code,2', 'Classification code'),
    ('Category)12,3', 'Category)'),
    ('Q3', 'Q3'), ('DJ54', 'DJ54'), ('13', '13'), ('2024', '2024'),
    ('1 - 10', '1 - 10'), ('250 - 999', '250 - 999'), ('13-15', '13-15'),
    ('Series2024', 'Series2024'),
    ('Đơn vị3', 'Đơn vị'),
])
def test_glued_footnotes(raw, expected):
    assert clean_text(raw) == expected
    assert clean_text(raw, False) == raw.strip()


@pytest.mark.parametrize('total', [
    'Total all categories7', 'TOTAL FOR firms(2)', 'All sizes',
    'Tổng số đơn vị3', 'Overall estimate', 'Grand total:', 'Total',
])
def test_summary_is_sibling_and_spaces_define_hierarchy(tmp_path, total):
    table = workbook(tmp_path, [
        ['Label', 'Mean7', 'Lower', 'Middle', 'Upper'],
        ['Example sector7', None, None, None, None],
        [total, 8.5, '(-8-)', '(-8-)', '(-8-)'],
        ['   2 - 20', 4.5, None, 3.5, 6.5],
        ['      Detail category', 2.5, 1.5, 2, 3],
        ['   21 - 40', 9.5, 5.5, 7.5, 10.5],
    ])
    frame = table.to_dataframe()
    summary = f'Example sector > {clean_text(total)}'
    assert frame.loc[summary, 'Mean'] == 8.5
    assert frame.loc[summary].iloc[1:].isna().all()
    detail = next(r for r in table.rows if r.label_path == 'Example sector > 2 - 20')
    assert detail.values == [4.5, None, 3.5, 6.5]
    assert 'Example sector > 2 - 20 > Detail category' in frame.index
    assert 'Example sector > 21 - 40' in frame.index
    assert table.rows[0].original_label == total


def test_stacked_fragments_offset_labels_sections_and_codes(tmp_path):
    table = workbook(tmp_path, [
        [1, 'Production estimates'],
        [None, None, None, 'Seasonally adjusted'],
        [None, None, 'Stone,', 'Medical'],
        [None, None, 'glass &', 'supplies,'],
        [None, None, 'other', 'Therapeut-'],
        [None, None, 'materials', 'ics'],
        ['Categories', None, '31-32', 45],
        ['Value (EUR million)'],
        [None, 2022, 11.5, 22.5],
        ['Index series'],
        [None, None, 'AB34', 'CD56'],
        [None, 2021, 81.5, 92.5],
        [None, 2022, 101.5, 112.5],
        [],
        [None, '2022 Q1', 102.5, 113.5],
        [None, 'Q2', 103.5, 114.5],
        ['Growth series'],
        [None, '2022 Q1', 1.5, 2.5],
        [None, 'Q2', 3.5, 4.5],
    ])
    assert table.header_paths == ['Stone, glass & other materials', 'Medical supplies, Therapeutics']
    assert not table.metadata['raw_fallback']
    frame = table.to_dataframe()
    assert len(frame) == 7
    assert frame.loc['Value (EUR million) > 2022'].tolist() == [11.5, 22.5]
    assert frame.loc['Index series > 2021'].tolist() == [81.5, 92.5]
    assert frame.loc['Index series > 2022 Q2'].tolist() == [103.5, 114.5]
    assert frame.loc['Growth series > 2022 Q2'].tolist() == [3.5, 4.5]
    prompt = build_prompt('Show production estimates', [table])[1]['content']
    assert 'Stone, glass & other materials' in prompt
    assert 'Index series > 2022 Q2' in prompt
    assert 'Column 3' not in prompt
