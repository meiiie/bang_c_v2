"""Deterministic formatting of computed answers."""
import math
import re
from numbers import Number
from decimal import Decimal, ROUND_HALF_UP, localcontext
from .parser import clean_text, normalize_value


def decimal_places(query) -> int | None:
    """Read an explicit decimal precision from common query wording."""
    english = dict(zip(
        ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
         'eight', 'nine', 'ten'), range(11)))
    count = r'(\d+|' + '|'.join(english) + r')'
    patterns = (
        (r'\b' + count + r'[\s-]+decimals?\b', english),
        (r'소수점\s*(첫째|둘째|\d+)\s*자리', {'첫째': 1, '둘째': 2}),
        (r'\b(\d+|một|hai)\s+chữ\s+số\s+thập\s+phân', {'một': 1, 'hai': 2}),
        (r'làm\s+tròn\s+đến\s+(\d+|một|hai)\s+chữ\s+số', {'một': 1, 'hai': 2}),
        (r'保留\s*(\d+|一|二|两)\s*位小数', {'一': 1, '二': 2, '两': 2}),
    )
    q = query.casefold()
    for pattern, words in patterns:
        match = re.search(pattern, q)
        if match:
            token = match.group(1)
            return int(token) if token.isdecimal() else words[token]
    # Preserve the legacy shorthand even without the word "decimal".
    if re.search(r'\brounded to two\b', q):
        return 2
    return None


def format_answer(raw, query):
    q = query.lower()
    if isinstance(raw, (list, tuple)):
        return ', '.join(format_answer(v, query) for v in raw)
    if isinstance(raw, str) and 'separated by a comma' in q and ',' in raw:
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        if len(parts) > 1:
            return format_answer(parts, query)
    if 'yes or no' in q:
        if isinstance(raw, str):
            v = raw.strip().lower()
            if v in ('yes', 'true', '1'): return 'Yes'
            if v in ('no', 'false', '0'): return 'No'
            raise ValueError('Expected boolean answer')
        return 'Yes' if bool(raw) else 'No'
    places = decimal_places(query)
    integer = 'integer' in q
    if isinstance(raw, str):
        raw = normalize_value(raw)
    if isinstance(raw, Number):
        if not math.isfinite(float(raw)): raise ValueError('Non-finite answer')
        number = Decimal(str(int(raw) if isinstance(raw, bool) else raw))
        if places is not None:
            with localcontext() as context:
                context.prec = max(context.prec, len(number.as_tuple().digits),
                                   number.adjusted() + places + 2)
                rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
            return format(rounded, f'.{places}f')
        if integer: return str(number.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        if float(raw).is_integer(): return str(int(raw))
        # Prefer the shortest nearby decimal, without erasing tiny values.
        for precision in range(1, 7):
            rounded = number.quantize(Decimal(1).scaleb(-precision), rounding=ROUND_HALF_UP)
            if rounded and abs(number - rounded) <= Decimal('1e-9'):
                return format(rounded, 'f').rstrip('0').rstrip('.')
    if raw is None: return ''
    text = str(clean_text(str(raw)))
    if ' > ' in text:
        text = text.split(' > ')[-1].strip()
    return _query_wording(text, query)


def _query_wording(text, query):
    """If the question itself names the candidates (e.g. 'Cedar, Orchid, and
    Maple'), echo the question's wording when the label is a longer form of it."""
    # A colon followed by a comma list is an explicit candidate list. Ordinary
    # prose mentioning a category must never shorten the returned table label.
    if ':' not in query:
        return text
    tail = query.split(':', 1)[1].split('.')[0]
    if ',' not in tail:
        return text
    names = [n.strip(" '\"") for n in re.split(r',\s*(?:and\s+|or\s+)?', tail)]
    names = [n for n in names if 0 < len(n) <= 60]
    low = text.casefold()
    matches = [n for n in names if low == n.casefold() or low.startswith(n.casefold() + ' ')]
    return matches[0] if len(matches) == 1 else text
