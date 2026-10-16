#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
test_dir="$(mktemp -d)"
trap 'rm -rf -- "$test_dir"' EXIT
python3 "$repo_dir/tools/build_zip.py"
python3 -m zipfile -e "$repo_dir/dist/submission.zip" "$test_dir/solution"
python3 -m venv "$test_dir/venv"
"$test_dir/venv/bin/python" -m pip install -r "$test_dir/solution/requirements.txt"
# Generate a disposable, wholly synthetic input; no contest files are needed.
data_dir="$test_dir/input"
"$test_dir/venv/bin/python" - "$data_dir" <<'PYDATA'
import json
import sys
from pathlib import Path
import openpyxl
root = Path(sys.argv[1])
(root / 'xlsx_files').mkdir(parents=True)
book = openpyxl.Workbook()
book.active.append(['Label', 'Count'])
book.active.append(['Synthetic item', 17])
book.save(root / 'xlsx_files' / 'tiny.xlsx')
book.close()
(root / 'queries.json').write_text(json.dumps([
    {'id': 'synthetic_1', 'file_name': 'tiny.xlsx',
     'query': 'What is the count for Synthetic item? Answer with only the integer.'}
]), encoding='utf-8')
PYDATA
cd "$test_dir/solution"
"$test_dir/venv/bin/python" main.py --input "$data_dir" --output dump.csv --mode dump
# An empty key prevents any network request, independent of the caller's environment.
OPENROUTER_API_KEY='' "$test_dir/venv/bin/python" main.py --input "$data_dir" --output fallback.csv --mode llm --profile contest --set api_key_env=OPENROUTER_API_KEY
"$test_dir/venv/bin/python" - "$data_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path
queries = json.loads((Path(sys.argv[1])/'queries.json').read_text(encoding='utf-8-sig'))
expected = [str(q['id']) for q in queries]
n = len(expected)
assert n >= 1 and all(expected) and len(set(expected)) == n
assert all(q['id'] is not None for q in queries)
prompt = Path('prompts/synthetic_1.txt').read_text(encoding='utf-8')
assert 'Synthetic item' in prompt and 'Count' in prompt
for name in ('dump.csv', 'fallback.csv'):
    with open(name, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['id', 'answer']
        rows = list(reader)
    assert [row['id'] for row in rows] == expected
    assert len(rows) == n
    assert all(row['answer'] == ('' if name == 'dump.csv' else '0') for row in rows)
print(f'Fresh install passed: dump and no-key fallback both have {n} unique IDs.')
PY
