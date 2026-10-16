"""Pure, deterministic heuristics for suspicious formatted Excel answers.

These checks flag assumptions for review; they never declare an answer wrong.
"""
import ast
from decimal import Decimal, InvalidOperation
from difflib import get_close_matches
import re
import unicodedata
from .formatter import format_answer


def normalize_label(value):
    text = unicodedata.normalize('NFC', str(value))
    text = re.sub(r'\(\d+\)|\[\d+\]|\s+\d+/|[*†‡]+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+', '', text)
    return re.sub(r'\s+', ' ', text).strip().casefold()


def table_labels(tables):
    """Include full selection paths and individual row/column label segments."""
    labels = []
    for table in tables:
        for path in [*table.header_paths, *(row.label_path for row in table.rows)]:
            labels.extend([str(path), *(part.strip() for part in str(path).split(' > '))])
    return list(dict.fromkeys(label for label in labels if label))


def closest_labels(value, labels, n=3):
    originals = {normalize_label(label): label for label in labels}
    matches = get_close_matches(normalize_label(value), sorted(originals), n=n, cutoff=0.25)
    return [originals[match] for match in matches]


def expected_item_count(question):
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    match = re.search(r'\b(\d+|' + '|'.join(words) + r')\s+(?:(?:\w+[ -]){0,3})'
                      r'(?:names|labels|integers|numbers|values|items)\b', question.casefold())
    if match:
        count = match.group(1)
        return int(count) if count.isdigit() else words[count]
    for template in re.findall(r'`([^`]+)`', question):
        parts = [part.strip() for part in template.split(',')]
        if len(parts) > 1 and all(re.fullmatch(r'[A-Za-z]\w*', part) for part in parts):
            return len(parts)
    return None


def _zero(value):
    try:
        return Decimal(str(value).strip()) == 0
    except InvalidOperation:
        return False


def missing_program_labels(code, labels):
    """Inspect literal string predicates without running model-generated code."""
    block = re.search(r'```(?:python)?\s*\n(.*?)```', code, re.S)
    try:
        tree = ast.parse(block.group(1) if block else code.strip())
    except SyntaxError:
        return []
    normalized = [normalize_label(label) for label in labels]
    missing = []
    for node in ast.walk(tree):
        predicates = []
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ('contains', 'startswith') and node.args:
                predicates.append((node.func.attr, node.args[0]))
        elif isinstance(node, ast.Compare):
            for left, op, right in zip([node.left, *node.comparators], node.ops, node.comparators):
                if isinstance(op, ast.Eq):
                    if not isinstance(left, ast.Constant): predicates.append(('==', right))
                    if not isinstance(right, ast.Constant): predicates.append(('==', left))
        for operation, literal in predicates:
            if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
                continue
            value = normalize_label(literal.value)
            # Substring/prefix predicates commonly select a segment of a path.
            exists = any(value in label if operation == 'contains' else
                         label.startswith(value) if operation == 'startswith' else
                         value == label for label in normalized)
            if value and not exists and literal.value not in missing:
                missing.append(literal.value)
    return missing


def check_answer(answer, question, labels, code='', raw=None):
    """Return concrete reasons and nearby table labels for a formatted candidate."""
    issues = []
    def add(reason, value):
        issues.append({'reason': reason, 'closest_labels': closest_labels(value, labels)})

    q = question.casefold()
    expected = expected_item_count(question)
    items = [format_answer(item, question) for item in raw] if isinstance(raw, (list, tuple)) else [answer]
    if not isinstance(raw, (list, tuple)) and expected is not None:
        items = [part.strip() for part in answer.split(',')]
    if re.search(r'\b(how many|count|total|sum|number of|combined)\b', q):
        zeros = items if len(items) > 1 else [part.strip() for part in answer.split(',')]
        if zeros and all(_zero(item) for item in zeros):
            add(f'Numeric answer is exactly zero (or an all-zero list): {answer!r}. A label filter may have matched no rows.', question)
    if expected is not None and len(items) != expected:
        add(f'Question requests {expected} items, but the answer has {len(items)}: {answer!r}.', question)
    if re.search(r'\bwhich\b|\bwhat is the name\b', q):
        known = {normalize_label(label) for label in labels}
        for item in items:
            value = str(item).strip()
            try:
                Decimal(value)
                continue
            except InvalidOperation:
                pass
            if normalize_label(value) not in known:
                add(f'Name/label answer {value!r} does not appear in any table row/column label segment.', value)
    for literal in missing_program_labels(code, labels):
        add(f'Program label predicate uses {literal!r}, which does not exist in the table labels.', literal)
    return issues


def retry_feedback(issues):
    details = '\n'.join(f"- {issue['reason']} Closest existing labels: {issue['closest_labels']!r}" for issue in issues)
    return ('Sanity check flagged the candidate:\n' + details + '\n'
            'Question your assumption: re-derive the row/column selection with exact labels from the table, possibly with a different approach.')
