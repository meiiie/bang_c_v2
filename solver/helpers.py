"""Single source of truth for prompt helper docs and worker functions."""
from dataclasses import dataclass
from typing import Callable
import re

def _find_label(labels, keywords, kind, section=None):
    from solver.parser import clean_text
    def cleaned(label):
        return clean_text(str(label)).casefold()
    def contains(label, keyword):
        return re.search(r'(?<!\w)' + re.escape(cleaned(keyword)) + r'(?!\w)', cleaned(label)) is not None
    labels = list(labels)
    matches = [label for label in labels
               if all(contains(label, word) for word in keywords)
               and (section is None or contains(label, section))]
    if len(matches) != 1:
        candidates = matches or [label for label in labels
                                if section is None or contains(label, section)] or labels
        raise ValueError(f'{kind} lookup expected one match for {keywords!r}, section={section!r}; '
                         f'found {len(matches)}. Candidates: {candidates!r}')
    return matches[0]


def find_row(df, *keywords, section=None):
    return _find_label(df.index, keywords, 'Row', section)


def find_col(df, *keywords):
    return _find_label(df.columns, keywords, 'Column')



def children(df, parent_label):
    """Return direct child rows as a DataFrame, retaining their full paths."""
    prefix = str(parent_label) + ' > '
    labels = [x for x in df.index if str(x).startswith(prefix) and ' > ' not in str(x)[len(prefix):]]
    return df.loc[labels]


def leaves(df_or_labels):
    labels = list(df_or_labels.index if hasattr(df_or_labels, 'columns') else df_or_labels)
    return [x for x in labels if not any(str(y).startswith(str(x) + ' > ') for y in labels)]


def age_at(df, birth_col, ref):
    import pandas as pd
    birth = pd.to_datetime(df[birth_col], errors='coerce')
    reference = pd.to_datetime(df[ref], errors='coerce') if ref in df.columns else pd.to_datetime(ref)
    return (reference - birth).dt.days


def sheet(tables, name_keyword):
    key = str(name_keyword).casefold()
    exact = [t for t in tables if key in (str(t.attrs.get('sheet_name', str(t.attrs.get('name', '')).rsplit(':', 1)[0])).casefold(), str(t.attrs.get('name', '')).casefold())]
    matches = exact or [t for t in tables if key in str(t.attrs.get('name', '')).casefold()]
    if len(matches) != 1:
        raise ValueError(f'Sheet lookup expected one match; candidates: {[t.attrs.get("name") for t in tables]}')
    return matches[0]


def to_number(x):
    import pandas as pd
    from solver.parser import normalize_value
    return pd.to_numeric(normalize_value(x), errors='coerce')


def parse_date(s, dayfirst=True):
    """Parse text dates explicitly; workbook cells themselves remain text."""
    import pandas as pd
    if isinstance(s, str) and re.fullmatch(r'\d{2}/\d{2}/\d{4}', s.strip()):
        from datetime import datetime
        return datetime.strptime(s.strip(), '%d/%m/%Y' if dayfirst else '%m/%d/%Y')
    return pd.to_datetime(s, dayfirst=dayfirst).to_pydatetime()



@dataclass(frozen=True)
class HelperSpec:
    name: str
    signature: str
    summary: str
    fn: Callable


HELPERS = [
    HelperSpec('find_row', 'find_row(df, *keywords, section=None)', 'Return one exact row label by cleaned case-insensitive keywords; ambiguous/missing matches raise with candidates.', find_row),
    HelperSpec('find_col', 'find_col(df, *keywords)', 'Return one exact column label by cleaned case-insensitive keywords; ambiguous/missing matches raise with candidates.', find_col),
    HelperSpec('children', 'children(df, parent_label)', 'Return a DataFrame of direct sub-rows under a full parent path.', children),
    HelperSpec('leaves', 'leaves(df_or_labels)', 'Return labels with no descendant paths (DataFrame index or list of row/column labels).', leaves),
    HelperSpec('age_at', 'age_at(df, birth_col, ref)', 'Return age in days; ref is an exact column name or date string.', age_at),
    HelperSpec('sheet', 'sheet(tables, name_keyword)', 'Pick one DataFrame, preferring exact sheet/table names before keyword containment; ambiguous matches raise.', sheet),
    HelperSpec('parse_date', 'parse_date(s, dayfirst=True)', 'Parse dates explicitly (dd/mm/yyyy by default); text date cells remain text.', parse_date),
    HelperSpec('to_number', 'to_number(x)', 'Convert a scalar to a number, handling commas and parenthesized negatives; missing/invalid values become NaN.', to_number),
]


def selected_specs(enabled=None):
    if enabled is None:
        return list(HELPERS)
    unknown = set(enabled) - {spec.name for spec in HELPERS}
    if unknown:
        raise ValueError(f'Unknown helpers: {sorted(unknown)}')
    return [spec for spec in HELPERS if spec.name in enabled]


def render_helpers(enabled=None):
    return '\n'.join(f'{s.signature}: {s.summary}' for s in selected_specs(enabled))


def helper_namespace(enabled=None):
    return {s.name: s.fn for s in selected_specs(enabled)}
