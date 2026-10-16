"""CLI for Excel QA; persist a complete fallback submission before solver setup."""
import argparse
import json
from pathlib import Path
import sys
import time
from solver.config import initial_rows, normalize_query_ids, write_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['llm', 'mock', 'dump'], default='llm')
    p.add_argument('--workers', type=int)
    p.add_argument('--dump-dir', default='prompts')
    p.add_argument('--mock-programs', default=str(Path(__file__).parent/'tests/oracle_programs'))
    p.add_argument('--budget', type=float)
    p.add_argument('--self-consistency', action='store_true')
    p.add_argument('--config')
    p.add_argument('--profile')
    p.add_argument('--checkpoint', metavar='PATH')
    p.add_argument('--no-checkpoint', action='store_true')
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = p.parse_args(argv)
    started = time.monotonic()
    # IDs must be readable to construct a submission. Workbook parsing happens later.
    queries = json.loads((Path(args.input)/'queries.json').read_text(encoding='utf-8-sig'))
    queries = normalize_query_ids(queries)
    rows = initial_rows(queries, args.mode)
    write_csv(args.output, rows)
    positions = {row['id']: i for i, row in enumerate(rows)}
    def update(row):
        rows[positions[str(row['id'])]] = row
        write_csv(args.output, rows)
    try:
        from solver.llm import LLMClient, MockClient
        from solver.pipeline import Pipeline
        from solver.config import load_config, apply_overrides
        config = load_config(args.config, args.profile)
        overrides = list(args.set)
        if args.workers is not None: overrides.append(f'workers={args.workers}')
        if args.budget is not None: overrides.append(f'budget={args.budget}')
        if args.self_consistency: overrides.append('strategy=vote')
        config = apply_overrides(config, overrides)
        if args.checkpoint is not None: config['checkpoint_path'] = args.checkpoint
        client = None if args.mode == 'llm' else MockClient(args.mock_programs)
        pipeline = Pipeline(client, config=config)
        pipeline.deadline = min(pipeline.deadline, started + config['total_time_limit'])
        pipeline.solve_all(queries, args.input, config['workers'], args.mode, args.dump_dir, on_answer=update,
                           output_path=args.output, checkpoint=not args.no_checkpoint, fresh=args.fresh,
                           client_factory=(lambda: LLMClient(config)) if args.mode == 'llm' else None)
        print(f'Log: {pipeline.log_path}')
    except BaseException as exc:
        print(f'Solver stopped; complete fallback CSV preserved: {type(exc).__name__}: {exc}', file=sys.stderr)
    finally:
        write_csv(args.output, rows)
    print(f'Wrote {len(rows)} answers to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
