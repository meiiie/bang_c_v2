#!/usr/bin/env python3
"""Summarize a questions.jsonl log and project 20 sequential questions.

Percentiles use linear interpolation. Token totals include all attempts (votes,
repairs and auxiliary calls); reasoning tokens are already part of completion
usage and are not added again. Missing usage is reported rather than estimated.
This is a latency projection, not the wall time of a concurrent run; unlogged
questions and run setup are not represented.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def _number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite nonnegative number')
    if integer and not isinstance(value, int):
        raise ValueError(f'{label} must be an integer')
    return value


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(path):
    records = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                _number(record['seconds'], 'seconds')
                _number(record['samples_used'], 'samples_used', integer=True)
                records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'line {line_number}: {exc}') from exc
    if not records:
        raise ValueError('No question records in log')
    seconds = sorted(record['seconds'] for record in records)
    prompt, completion, missing, usage_attempts = 0, 0, 0, 0
    for record in records:
        for attempt in record.get('attempts', []):
            usage = attempt.get('usage') or {}
            known = True
            for field in ('prompt_tokens', 'completion_tokens'):
                value = usage.get(field)
                if value is None:
                    known = False
                else:
                    value = _number(value, field, integer=True)
                    if field == 'prompt_tokens':
                        prompt += value
                    else:
                        completion += value
            usage_attempts += 1
            missing += int(not known)
    mean = statistics.mean(seconds)
    projected = 20 * mean
    return dict(questions=len(records), sequential_seconds=sum(seconds), mean_seconds=mean,
                p50_seconds=_percentile(seconds, .5), p90_seconds=_percentile(seconds, .9),
                max_seconds=seconds[-1], samples_used_distribution=dict(sorted(Counter(
                    record['samples_used'] for record in records).items())),
                prompt_tokens_per_question=prompt / len(records),
                completion_tokens_per_question=completion / len(records),
                tokens_per_question=(prompt + completion) / len(records),
                usage_attempts=usage_attempts, usage_attempts_missing=missing,
                projected_20_seconds=projected, guard_seconds=2100,
                guard_headroom_seconds=2100 - projected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path, help='runs/<id>/questions.jsonl')
    parser.add_argument('--json', action='store_true', help='emit machine-readable metrics')
    args = parser.parse_args(argv)
    try:
        report = summarize(args.log)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"Questions logged: {report['questions']}")
        print(f"Sequential-equivalent total: {report['sequential_seconds']:.2f} s")
        print('Latency (s): ' + ', '.join(f"{name}={report[key]:.2f}" for name, key in (
            ('mean', 'mean_seconds'), ('p50', 'p50_seconds'), ('p90', 'p90_seconds'), ('max', 'max_seconds'))))
        print(f"Samples-used distribution: {report['samples_used_distribution']}")
        print(f"Known tokens/q: {report['tokens_per_question']:.2f} "
              f"(prompt={report['prompt_tokens_per_question']:.2f}, "
              f"completion={report['completion_tokens_per_question']:.2f})")
        print(f"Attempts with missing token usage: {report['usage_attempts_missing']}/{report['usage_attempts']}")
        print(f"20 questions one at a time: {report['projected_20_seconds']:.2f} s "
              f"({report['projected_20_seconds'] / 60:.2f} min) vs 35-min guard (2100 s)")
        headroom = report['guard_headroom_seconds']
        print(f"{'Headroom' if headroom >= 0 else 'Over guard'}: {abs(headroom):.2f} s")
        print('Projection uses logged question latencies; excludes setup and unlogged questions.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
