#!/usr/bin/env python3
"""Resolve container CLI paths and run the solver from writable scratch space."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import NamedTuple


class Resolved(NamedTuple):
    input: Path
    output: Path
    argv: list[str]
    env: dict[str, str]


def resolve(argv=None, environ=None):
    """Resolve flags/positionals without changing cwd or mutating the environment."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input', '-i')
    parser.add_argument('--output', '-o')
    parser.add_argument('paths', nargs='*', metavar='INPUT_DIR [OUTPUT]')
    # Mirror main.py's options so their values cannot be mistaken for positionals.
    for flag in ('profile', 'mode', 'workers', 'dump-dir', 'mock-programs',
                 'budget', 'config', 'checkpoint'):
        parser.add_argument('--' + flag)
    for flag in ('self-consistency', 'no-checkpoint', 'fresh'):
        parser.add_argument('--' + flag, action='store_true')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_intermixed_args(argv)
    paths = iter(args.paths)
    input_path = args.input or next(paths, None)
    output_path = args.output or next(paths, None)
    if next(paths, None) is not None:
        parser.error('too many positional paths')
    if input_path is None:
        input_path = ('/data' if Path('/data/queries.json').is_file()
                      else '/input' if Path('/input').exists() else '/data')
    output = Path(output_path or '/output/submission.csv').expanduser()
    if output.is_dir() or output.suffix.lower() != '.csv' or str(output_path).endswith('/'):
        output /= 'submission.csv'
    input_dir, output = Path(input_path).expanduser().resolve(), output.resolve()
    forwarded = ['--input', str(input_dir), '--output', str(output)]
    for flag in ('profile', 'mode', 'workers', 'dump-dir', 'mock-programs',
                 'budget', 'config', 'checkpoint'):
        value = getattr(args, flag.replace('-', '_'))
        if value is None:
            value = {'profile': 'contest', 'mode': 'llm'}.get(flag)
        if value is not None:
            # These files are read after switching to scratch; retain caller paths.
            if flag in ('config', 'mock-programs'):
                value = str(Path(value).expanduser().resolve())
            forwarded.extend(['--' + flag, value])
    for flag in ('self-consistency', 'no-checkpoint', 'fresh'):
        if getattr(args, flag.replace('-', '_')):
            forwarded.append('--' + flag)
    for value in args.set:
        forwarded.extend(['--set', value])
    env = dict(os.environ if environ is None else environ)
    if 'OPENROUTER_API_KEY' not in env:
        for alias in ('OPENROUTER_KEY', 'OPENROUTER_API', 'API_KEY'):
            if env.get(alias):
                env['OPENROUTER_API_KEY'] = env[alias]
                break
    return Resolved(input_dir, output, forwarded, env)


def main(argv=None):
    resolved = resolve(argv)
    created_output_dir = False
    try:
        resolved.output.parent.mkdir(parents=True)
        created_output_dir = True
    except FileExistsError:
        if not resolved.output.parent.is_dir():
            raise
    print(f'Input: {resolved.input}\nOutput: {resolved.output}', flush=True)
    # runs/, default dump prompts, relative checkpoint overrides and temp files
    # all stay writable even when /app is mounted read-only. Default checkpoints
    # remain beside the output CSV, allowing resume across container invocations.
    with tempfile.TemporaryDirectory(prefix='excel-qa-') as scratch:
        print(f'Scratch: {scratch}', flush=True)
        env = dict(resolved.env, TMPDIR=scratch)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().with_name('main.py')), *resolved.argv],
            cwd=scratch, env=env,
        )
    for path, mode in ([(resolved.output, 0o644)] +
                       ([(resolved.output.parent, 0o755)] if created_output_dir else [])):
        try:
            path.chmod(mode)
        except OSError:
            pass
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
