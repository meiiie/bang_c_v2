"""Exact-match CSV evaluation."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if any('id' not in r or 'answer' not in r for r in rows): raise ValueError('Expected id,answer columns')
    result = {r['id'].strip(): r['answer'].strip() for r in rows}
    if len(result) != len(rows): raise ValueError('Duplicate IDs')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred', required=True)
    p.add_argument('--gold', required=True)
    p.add_argument('--queries', help='Query JSON with optional type fields; defaults to queries.json beside --gold')
    a = p.parse_args()
    pred, gold = read_csv(a.pred), read_csv(a.gold)
    query_path = Path(a.queries) if a.queries else Path(a.gold).parent / 'queries.json'
    query_types = {}
    if a.queries or query_path.is_file():
        queries = json.loads(query_path.read_text(encoding='utf-8-sig'))
        for query in queries:
            if query['id'] in query_types:
                raise ValueError('Duplicate query IDs')
            query_types[query['id']] = query.get('type')
    typed = any(query_types.get(key) for key in gold)
    per_type = defaultdict(lambda: [0, 0])
    correct = 0
    for key, answer in gold.items():
        value = pred.get(key)
        match = value == answer
        correct += match
        if typed:
            kind = query_types.get(key) or 'untyped'
            per_type[kind][0] += match
            per_type[kind][1] += 1
        print(f'{key}: OK' if match else f'{key}: predicted={value!r} gold={answer!r}')
    for key in pred.keys()-gold.keys(): print(f'Unexpected ID: {key}')
    print(f'Accuracy: {correct}/{len(gold)} ({correct/len(gold):.2%})' if gold else 'Accuracy: 0/0')
    for kind, (hits, total) in sorted(per_type.items()):
        print(f'Type {kind}: {hits}/{total} ({hits/total:.2%})')


if __name__ == '__main__': main()
