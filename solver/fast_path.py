"""Conservative exact-label single-cell lookup; no fuzzy matching or arithmetic."""
import re
from .formatter import format_answer


def lookup(question, tables):
    # The entire question body must match a lookup grammar. Extra qualifications,
    # arithmetic, and implicit demographic sections fall through to the model.
    body = question.split('?')[0].strip()
    if '?' not in question: return None
    suffix = question.split('?', 1)[1].strip()
    if suffix and not re.fullmatch(r'Answer with only (?:the integer, without commas or units|the number rounded to two decimal places, without units|the value)\.', suffix):
        return None
    match = re.fullmatch(r'What is (?:the number of |the value of |the )?(.+?) (?:in|for) (.+)', body)
    if not match: return None
    def unquote(text):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in '\"\'':
            return text[1:-1]
        return text
    column, row = map(unquote, match.groups())
    # Match whole labels (or unique exact terminal segments) globally. Repeated
    # labels in sections/tables are deliberately ambiguous even if values agree.
    rows = [(t, r) for t in tables for r in t.rows
            if row in (r.label_path, r.label_path.split(' > ')[-1])]
    cols = [(t, i) for t in tables for i, c in enumerate(t.header_paths)
            if column in (c, c.split(' > ')[-1])]
    if len(rows) != 1 or len(cols) != 1 or rows[0][0] is not cols[0][0]: return None
    value = rows[0][1].values[cols[0][1]]
    if value is None or value == '': return None
    return format_answer(value, question)
