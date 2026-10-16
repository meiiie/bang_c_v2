from decimal import Decimal

import pytest
from solver.formatter import format_answer


@pytest.mark.parametrize('raw,query,expected', [
    (True,'Yes or No','Yes'), (False,'Yes or No','No'), ('no','Yes or No','No'),
    (1234.0,'integer','1234'), (12.6,'integer','13'), ('1,234','integer','1234'),
    (7.82641,'rounded to two decimal places','7.83'),
    ([21.8,16.2],'exactly two decimal places','21.80, 16.20'),
    ((24680,13579),'two integers','24680, 13579'),
    (['Orchid','Maple','Cedar'],'group names','Orchid, Maple, Cedar'),
    ('  Total(2)\n ','name','Total'), ('Net inventory 3/','name','Net inventory'),
    (None,'name',''),
])
def test_format(raw,query,expected):
    assert format_answer(raw,query) == expected


def test_invalid_number():
    with pytest.raises(ValueError): format_answer(float('nan'),'integer')


def test_path_and_query_wording():
    from solver.formatter import format_answer
    q = "Rank the following groups by X: Cedar, Orchid, and Maple. Answer with names."
    assert format_answer(['Orchid', 'Maple', 'Cedar and related'], q) == 'Orchid, Maple, Cedar'
    assert format_answer('Catalog > Garden tools > Copper spades', 'Which category? Answer with only the category name.') == 'Copper spades'
    assert format_answer('Research and design', 'Identify the industry with the highest number. Answer with only the industry name.') == 'Research and design'


def test_comma_string_answer_is_split_and_normalized():
    from solver.formatter import format_answer
    q = "Rank these: Cedar, Orchid, and Maple. Answer with the names, separated by a comma and one space."
    assert format_answer('Orchid, Maple, Cedar and related', q) == 'Orchid, Maple, Cedar'
    assert format_answer('24680, 13579', 'Answer as two integers, separated by a comma and one space, without thousands separators.') == '24680, 13579'


@pytest.mark.parametrize('raw', [4.699999999999999, 4.7, Decimal('4.7')])
def test_exactly_one_decimal_place(raw):
    assert format_answer(raw, 'Answer with only the number using exactly one decimal place.') == '4.7'


@pytest.mark.parametrize('query,expected', [
    ('one decimal', '4.7'),
    ('1 decimal', '4.7'),
    ('1-decimal place', '4.7'),
    ('rounded to 3 decimal places', '4.700'),
    ('exactly four decimal places', '4.7000'),
    ('12 decimal places', '4.700000000000'),
    ('two decimal places', '4.70'),
    ('rounded to two', '4.70'),
    ('2-decimal places', '4.70'),
    ('2 decimals', '4.70'),
    ('소수점 둘째 자리', '4.70'),
    ('소수점 첫째 자리', '4.7'),
    ('một chữ số thập phân', '4.7'),
    ('hai chữ số thập phân', '4.70'),
    ('làm tròn đến 3 chữ số', '4.700'),
    ('保留一位小数', '4.7'),
    ('保留两位小数', '4.70'),
    ('0 decimal places', '5'),
])
def test_requested_decimal_precision(query, expected):
    assert format_answer(4.7, query) == expected


@pytest.mark.parametrize('raw,query,expected', [
    (Decimal('4.65'), 'one decimal place', '4.7'),
    (Decimal('-4.65'), 'one decimal place', '-4.7'),
    (Decimal('4.665'), 'two decimal places', '4.67'),
    (4.699999999999999, 'Answer with only the number.', '4.7'),
    (0.1 + 0.2, 'Answer with only the number.', '0.3'),
    (1.2345670000000002, 'number', '1.234567'),
    (1.23456789, 'number', '1.23456789'),
    (1e-12, 'number', '1e-12'),
    (4.699999999999999, 'integer', '5'),
    (True, 'Yes or No, using one decimal place', 'Yes'),
])
def test_decimal_rounding_and_numeric_fallback(raw, query, expected):
    assert format_answer(raw, query) == expected
