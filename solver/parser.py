"""Parse spreadsheet layouts without workbook-specific rules."""
from dataclasses import dataclass, field
from datetime import datetime, date
from numbers import Number
import re
import openpyxl


def clean_text(value, strip_footnotes=True):
    if not isinstance(value, str):
        return value
    value = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])', '', value)
    value = re.sub(r'\s+', ' ', value).strip()
    if strip_footnotes:
        value = re.sub(r'\(\d+\)|\s+\d+/', '', value).strip()
        value = re.sub(r'([^\W\d_]|\)),?\d{1,2}(?:,\d{1,2})*$',
                       lambda m: m[1] if m[1].islower() or m[1] == ')' else m[0],
                       value).strip()
    return value


_NUMBER = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')


def is_missing_marker(value):
    """Recognize placeholders without confusing them with negative numbers."""
    if not isinstance(value, str):
        return False
    text = clean_text(value, False).replace('−', '-')
    return (text.casefold() in {'-', '–', '—', 'na', 'n/a', 'n.a.'}
            or re.fullmatch(r'\(\s*[-–—]\s*8\s*[-–—]\s*\)', text) is not None)


def normalize_value(value):
    """Percent strings store percentage points (12.5%, not 0.125).

    Dates remain text; parsing them requires an explicit day-first helper.
    """
    if not isinstance(value, str):
        return value
    text = value.strip().replace('−', '-')
    if is_missing_marker(text):
        return None
    percent = text.endswith('%')
    if percent:
        text = text[:-1].strip()
    negative = text.startswith('(') and text.endswith(')')
    if negative:
        text = text[1:-1].strip()
    if not _NUMBER.fullmatch(text):
        return value
    text = text.replace(',', '')
    number = float(text) if any(c in text for c in '.eE') else int(text)
    return -number if negative else number


class GridRow(list):
    """Carry source indentation, missing markers and percent units through stages."""
    def __init__(self, values, indents=None, percent_cols=None):
        super().__init__(values)
        self.indents = indents or [0] * len(self)
        self.percent_cols = set() if percent_cols is None else percent_cols
        self.missing_cols = set()


@dataclass
class Row:
    label_path: str
    values: list
    original_label: str = ''


@dataclass
class Table:
    name: str
    header_paths: list[str]
    rows: list[Row]
    units: str = ''
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self):
        import pandas as pd
        frame = pd.DataFrame([r.values for r in self.rows], columns=self.header_paths,
                             index=[r.label_path for r in self.rows])
        frame.attrs.update(self.metadata, units=self.units, name=self.name)
        return frame

    def to_text(self, max_chars=16000, row_indices=None, serialization="pipe"):
        if serialization == "compact":
            return self.to_compact(max_chars, row_indices)
        def render(v):
            if v is None:
                return ''
            if isinstance(v, (datetime, date)):
                return v.isoformat()
            if isinstance(v, float):
                return format(v, '.12g')
            return str(v).replace('|', '/')
        lines = [f'Table: {self.name}; units: {self.units}' + ('; Notes: ' + ' | '.join(self.metadata.get('notes', [])) if self.metadata.get('notes') else ''),
                 'Columns: ' + repr(self.header_paths),
                 'Column positions: ' + repr(list(range(len(self.header_paths)))),
                 'Rows: iloc position | exact DataFrame index string | values in column-list order']
        indices = list(range(len(self.rows))) if row_indices is None else row_indices
        for i in indices:
            r = self.rows[i]
            line = f'{i} | {r.label_path!r} | ' + ' | '.join(render(v) for v in r.values)
            if len('\n'.join(lines)) + len(line) + 1 > max_chars:
                break
            lines.append(line)
        if len(lines) - 4 < len(self.rows):
            note = '\n[Remaining rows omitted from prompt; all rows available in tables.]'
        else:
            note = ''
        return ('\n'.join(lines) + note)[:max_chars]


    def to_compact(self, max_chars=16000, row_indices=None):
        paths = [p.split(' > ') for p in self.header_paths]
        common = []
        for parts in zip(*paths):
            if len(set(parts)) != 1 or len(common) >= min(map(len, paths)) - 1:
                break
            common.append(parts[0])
        prefix = ' > '.join(common) + ' > ' if common else ''
        lines = [f'Table: {self.name}; units: {self.units}' + ('; Notes: ' + ' | '.join(self.metadata.get('notes', [])) if self.metadata.get('notes') else ''),
                 'Columns: ' + repr(self.header_paths),
                 'Table column prefix: ' + repr(prefix),
                 'Header ids: ' + repr(list(range(len(self.header_paths)))),
                 'Rows: exact index path | values by header id; trailing empties omitted']
        used = 0
        for i in range(len(self.rows)) if row_indices is None else row_indices:
            row = self.rows[i]
            values = list(row.values)
            while values and values[-1] in (None, ''):
                values.pop()
            def render(v):
                if v is None: return ''
                if isinstance(v, (datetime, date)): return v.isoformat()
                if isinstance(v, float): return format(v, '.12g')
                return str(v).replace('|', '/')
            line = row.label_path + ' | ' + ' | '.join(map(render, values))
            if len('\n'.join(lines)) + len(line) + 1 > max_chars: break
            lines.append(line)
            used += 1
        if used < len(self.rows):
            lines.append('[Remaining rows omitted from prompt; all rows available in tables.]')
        return '\n'.join(lines)[:max_chars]


def is_data(row):
    if row[0] in (None, ''):
        return False
    vals = [v for v in map(normalize_value, row[1:]) if v not in (None, '')]
    if vals and all(isinstance(v, int) and not isinstance(v, bool) and 1900 <= v <= 2100 for v in vals):
        return False
    return bool(vals) and sum(isinstance(v, (Number, datetime, date)) for v in vals) >= len(vals)/2


_BARE_TOTAL = re.compile(r'^(?:grand\s+total|total|all\s+sizes|tổng|overall|cộng)\b', re.I)
_SIZE_CLASS = re.compile(r'^\d[\d,]*\s*(?:[-–—−]\s*\d[\d,]*|\+)$')


def _label_column(block):
    if block and block[0][0] == 'Indent Level':
        return 1
    # A leading section column can be empty on observations. Require several
    # numeric measures to avoid mistaking ordinary sparse headers for records.
    observations = [r for r in block if len(r) > 2 and r[1] not in (None, '')
                    and sum(isinstance(normalize_value(v), Number) for v in r[2:]) >= 2]
    if observations and sum(r[0] in (None, '') for r in observations) > len(observations) / 2:
        return 1
    return 0


def _fragment_headers(rows, label_col, columns):
    """Find a stacked text band with no labels in the leading columns.

    Ordinary merged header hierarchies retain their path separators; a band
    of at least three unlabeled text rows represents wrapped column prose.
    """
    band = []
    for row in rows:
        values = [row[c] for c in columns if row[c] not in (None, '')]
        if (all(v in (None, '') for v in row[:label_col+1])
                and len(values) >= max(2, len(columns) / 2)
                and all(isinstance(v, str) and re.search(r'[a-z]', v) for v in values)):
            band.append(row)
        else:
            if len(band) >= 3:
                return band
            band = []
    return band if len(band) >= 3 else []


def load_grid(state):
    cached = openpyxl.load_workbook(state["path"], data_only=True)
    formulas = openpyxl.load_workbook(state["path"], data_only=False)
    sheets = []
    try:
        for sheet in cached:
            source = formulas[sheet.title]
            def resolve(coord, seen=None):
                seen = set() if seen is None else seen
                if coord in seen:
                    return None
                seen.add(coord)
                v = sheet[coord].value
                if v is None:
                    f = source[coord].value
                    if isinstance(f, str) and re.fullmatch(r'=\$?[A-Z]+\$?\d+', f):
                        return resolve(f[1:].replace('$', ''), seen)
                return v
            raw = [[resolve(c.coordinate) for c in row] for row in sheet]
            occupied = [i for row in raw for i, v in enumerate(row) if v not in (None, '')]
            if not occupied:
                continue
            width = max(occupied) + 1
            grid = []
            for values, cells in zip(raw, sheet):
                indents = [float(c.alignment.indent or 0) +
                           (len(c.value) - len(c.value.lstrip()) if isinstance(c.value, str) else 0)
                           for c in cells[:width]]
                grid.append(GridRow(values[:width], indents))
            sheets.append(dict(name=sheet.title, grid=grid, raw_grid=[GridRow(r, r.indents[:]) for r in grid], width=width, merges=list(sheet.merged_cells.ranges)))
    finally:
        cached.close()
        formulas.close()
    state["sheets"] = sheets
    return state

def _unmerge(state):
    grid = state['grid']
    width = state['width']
    merges = state['merges']
    full_rows = set()
    for area in merges:
        value = grid[area.min_row-1][area.min_col-1]
        if area.min_col == 1 and area.max_col >= width:
            full_rows.update(range(area.min_row-1, area.max_row))
        for r in range(area.min_row-1, area.max_row):
            for c in range(area.min_col-1, min(area.max_col, width)):
                grid[r][c] = value

    state.update(grid=grid, full_rows=full_rows)
    return state

def _drop(state):
    grid = state['grid']
    full_rows = state['full_rows']
    units = ''
    original_text = {clean_text(v): clean_text(v, False) for row in grid for v in row if isinstance(v, str)}
    filtered, notes = [], []
    for i, row in enumerate(grid):
        text = clean_text(row[0]) or ''
        if isinstance(text, str) and (text.startswith('Footnotes:') or text.startswith('NOTE:') or re.match(r'^\(\d+\) ', text)):
            break
        if isinstance(text, str) and text.startswith('['):
            units = text.strip('[]')
            continue
        if i in full_rows:
            if text:
                notes.append(clean_text(text, False))
            continue
        if any(isinstance(v, str) and v.lower() in ('billion dollars', 'million dollars') for v in row):
            units = next(v for v in row if isinstance(v, str) and 'dollars' in v.lower())
            continue
        filtered.append(GridRow([clean_text(v, False) for v in row], getattr(row, 'indents', None)))

    state.update(filtered=filtered, units=units, original_text=original_text, notes=notes)
    return state


def normalize_values(state):
    for sheet in state.get('sheets', []):
        for row in sheet.get('filtered', sheet['grid']):
            percents = set()
            missing = set()
            # Column zero may be an explicit indent level; textual labels are
            # deliberately preserved, including numeric IDs and footnotes.
            for c in range(1, len(row)):
                original = row[c]
                if is_missing_marker(original):
                    missing.add(c)
                row[c] = normalize_value(original)
                if isinstance(original, str) and original.strip().endswith('%') and isinstance(row[c], Number):
                    percents.add(c)
            if isinstance(row, GridRow):
                row.percent_cols = percents
                row.missing_cols = missing
    return state


def _split(state):
    filtered = state['filtered']
    label_col = _label_column(filtered)
    # Blank rows alone do not split demographic sections. A new column
    # block has multiple populated cells and is not a labeled data row.
    blocks, block, data_seen, blank = [], [], False, False
    for row in filtered:
        if not any(v not in (None, '') for v in row):
            blank = True
            continue
        data = is_data(row[label_col:])
        if data_seen and blank and not data and sum(v not in (None, '') for v in row[label_col+1:]) >= 2:
            blocks.append(block)
            block, data_seen = [], False
        block.append(row)
        data_seen = data_seen or data
        blank = False
    if block:
        blocks.append(block)

    state.update(blocks=blocks)
    return state

def detect_headers(state):
    records = []
    for s in state["sheets"]:
        width = s["width"]
        for b, block in enumerate(s.get("blocks", [s.get("filtered", s["grid"])])):
            label_col = _label_column(block)
            first = next((i for i, r in enumerate(block) if is_data(r[label_col:])), None)
            if first is None:
                continue
            columns = list(range(label_col+1, width))
            headers = []
            header_rows = [r for r in block[:first] if sum(v not in (None, '') for v in r[label_col+1:]) > 0]
            fragments = _fragment_headers(header_rows, label_col, columns)
            if fragments:
                header_rows = fragments
            for c in columns:
                parts = []
                for row in header_rows:
                    v = clean_text(row[c])
                    if v not in (None, '') and (fragments or str(v) not in parts):
                        parts.append(str(v))
                if fragments:
                    header = '\n'.join(parts)
                    header = re.sub(r'(?<=[A-Za-z])-\n(?=[a-z])', '', header)
                    headers.append(clean_text(header) or f'Column {c+1}')
                else:
                    headers.append(' > '.join(parts) or f'Column {c+1}')
            for i, c in enumerate(columns):
                populated = [row for row in block[first:] if row[label_col] not in (None, '') and row[c] not in (None, '')]
                if populated and all(c in getattr(row, 'percent_cols', set()) for row in populated):
                    headers[i] += ' (%)'
            records.append(dict(s, b=b, block=block, first=first, label_col=label_col, columns=columns, headers=headers))
        if not any(r['name'] == s['name'] for r in records):
            raw = s.get('raw_grid', s['grid'])
            nonempty = [r for r in raw if any(v not in (None, '') for v in r)]
            headers = [str(clean_text(v)) if v not in (None, '') else f'Column {c+1}' for c, v in enumerate(nonempty[0])]
            records.append(dict(s, b=0, block=nonempty, first=1, label_col=0, columns=list(range(width)), headers=headers, raw_fallback=True))
    state["blocks"] = records
    return state

def build_row_paths(state):
    result = []
    for s in state.get("blocks", []):
        name = s['name']
        b = s['b']
        block = s['block']
        first = s['first']
        label_col = s['label_col']
        columns = s['columns']
        headers = s['headers']
        units = s['units']
        original_text = s['original_text']
        rows, sections, indent_stack = [], [], {}
        group = ''
        explicit_indent = block[0][0] == 'Indent Level'
        offset_labels = label_col > 0 and not explicit_indent
        use_indent = explicit_indent or any(getattr(r, 'indents', [0]*len(r))[label_col] for r in block)
        quarter_year = None
        section_start = first
        while section_start > 0 and not any(v not in (None, '') for v in block[section_start-1][label_col+1:]):
            section_start -= 1
        for row in block[section_start:]:
            if s.get('raw_fallback'):
                rows.append(Row(f'Row {len(rows)+1}', [normalize_value(row[c]) for c in columns]))
                continue
            label = row[label_col]
            if offset_labels and label in (None, '') and not any(v not in (None, '') for v in row[label_col+1:]):
                label = next((v for v in row[:label_col] if v not in (None, '')), None)
            if label in (None, ''):
                continue
            original = str(label)
            label = clean_text(original)
            values = [row[c] for c in columns]
            if not any(v not in (None, '') for v in values):
                # Missing observations and empty size classes are detail rows,
                # so they must not replace the current industry section.
                if (_SIZE_CLASS.fullmatch(clean_text(label))
                        or any(c in getattr(row, 'missing_cols', set()) for c in columns)):
                    continue
                if label.isupper():
                    sections = [label]
                else:
                    sections = ([sections[0]] if sections and sections[0].isupper() and sections[0] != 'TOTAL' else []) + [label]
                group = ''
                indent_stack = {}
                quarter_year = None
                continue
            if offset_labels:
                period = re.fullmatch(r'(\d{4})\s+Q[1-4]', label, re.I)
                if period:
                    quarter_year = period[1]
                elif quarter_year and re.fullmatch(r'Q[1-4]', label, re.I):
                    label = quarter_year + ' ' + label
            if use_indent:
                level = int(row[0] or 0) if explicit_indent else getattr(row, 'indents', [0]*len(row))[label_col]
                indent_stack = {k: v for k, v in indent_stack.items() if k < level}
                prefix = sections + [indent_stack[k] for k in sorted(indent_stack)]
                # Total-like labels summarise their siblings;
                # it must never become the path parent of the detail rows below it.
                if not _BARE_TOTAL.match(label.strip()):
                    indent_stack[level] = label
            else:
                if ', 16 years and over' in label and not label.startswith('Total,'):
                    group = label
                    prefix = sections
                else:
                    prefix = sections + ([group] if group else [])
            rows.append(Row(' > '.join(prefix + [label]), values, original_text.get(original, original)))
        result.append(Table(f'{name}:{b+1}', headers, rows, units,
                            {'sheet_name': name, 'notes': s.get('notes', []), 'raw_fallback': s.get('raw_fallback', False), 'label_header': str(block[0][label_col] or ''),
                             'original_headers': [original_text.get(h.split(' > ')[-1], h) for h in headers]}))
    state["tables"] = result
    return state

def _each_sheet(state, fn):
    for s in state.get('sheets', []):
        s.setdefault('full_rows', set())
        s.setdefault('filtered', s['grid'])
        s.setdefault('units', '')
        s.setdefault('original_text', {})
        fn(s)
    return state


def unmerge_fill(state):
    return _each_sheet(state, _unmerge)


def drop_titles_footnotes(state):
    return _each_sheet(state, _drop)


def split_tables(state):
    return _each_sheet(state, _split)


def clean_labels(state):
    for t in state.get('tables', []):
        t.header_paths = [' > '.join(clean_text(p) for p in h.split(' > ')) for h in t.header_paths]
        for r in t.rows:
            r.label_path = ' > '.join(clean_text(p) for p in r.label_path.split(' > '))
            r.values = [clean_text(v, False) for v in r.values]
        # All distinguishing ancestors are retained above. If the same full
        # path still repeats (including footnote-only differences), number it.
        reserved = {r.label_path for r in t.rows}
        used = set()
        for r in t.rows:
            base, n = r.label_path, 2
            while r.label_path in used:
                r.label_path = f'{base} #{n}'
                n += 1
                if r.label_path in reserved:
                    r.label_path = base
            used.add(r.label_path)
    return state


def _add_age_columns(table):
    """When a record table has a birth-date column and other date columns, add
    'Age at <event> (days)' so age rankings are measured at the record's own date."""
    def is_dates(i):
        vals = [r.values[i] for r in table.rows if r.values[i] not in (None, '')]
        return bool(vals) and all(isinstance(v, (datetime, date)) for v in vals)
    dates = [i for i in range(len(table.header_paths)) if is_dates(i)]
    births = [i for i in dates if 'birth' in table.header_paths[i].lower()]
    if len(births) != 1:
        return table
    b = births[0]
    for i in dates:
        if i == b:
            continue
        event = table.header_paths[i].split(' > ')[-1]
        table.header_paths.append(f'Age at {event} (days)')
        for r in table.rows:
            x, y = r.values[i], r.values[b]
            r.values.append((x - y).days if isinstance(x, (datetime, date)) and isinstance(y, (datetime, date)) else None)
    return table


ENRICHERS = {'age_at_event': _add_age_columns}


def enrichers(state):
    for name in state['config'].get('enrichers', []):
        for table in state.get('tables', []):
            ENRICHERS[name](table)
    return state


STAGES = [load_grid, unmerge_fill, drop_titles_footnotes, normalize_values, split_tables,
          detect_headers, build_row_paths, clean_labels, enrichers]


def parse_workbook(path, config=None):
    from .config import load_config
    config = load_config() if config is None else config
    enabled = config['stages']
    unknown = set(enabled) - {fn.__name__ for fn in STAGES}
    if unknown:
        raise ValueError(f'Unknown parser stages: {sorted(unknown)}')
    state = {'path': path, 'config': config}
    for fn in STAGES:
        if fn.__name__ in enabled:
            state = fn(state)
    return state.get('tables', [])
