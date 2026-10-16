"""Container CLI and writable-scratch checks, runnable with pytest or unittest."""
import csv
import importlib.util
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('docker_entrypoint', ROOT/'docker-entrypoint.py')
entrypoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entrypoint)
resolve = entrypoint.resolve


class DockerEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def test_flags_and_positionals(self):
        for flags in (
            ['--input', 'data', '--output', 'answers.csv'],
            ['-i', 'data', '-o', 'answers.csv'],
            ['--input=data', '--output=answers.csv'],
            ['data', 'answers.csv'],
            ['data', '--output', 'answers.csv'],
        ):
            with self.subTest(flags=flags):
                result = resolve(flags, {})
                self.assertEqual(result.input, self.root/'data')
                self.assertEqual(result.output, self.root/'answers.csv')
                self.assertEqual(result.argv[:4], ['--input', str(result.input), '--output', str(result.output)])

    def test_output_directory_or_csv(self):
        (self.root/'existing.csv').mkdir()
        for output in ('new-dir', 'new-dir/', 'answers.csv/', 'existing.csv', 'answers.csv'):
            with self.subTest(output=output):
                expected = self.root/output
                if output != 'answers.csv':
                    expected /= 'submission.csv'
                self.assertEqual(resolve(['data', output], {}).output, expected)

    def test_defaults(self):
        for data_queries, input_exists, expected in (
            (True, True, '/data'), (True, False, '/data'),
            (False, True, '/input'), (False, False, '/data'),
        ):
            with self.subTest(data_queries=data_queries, input_exists=input_exists):
                with patch.object(Path, 'is_file', lambda p: str(p) == '/data/queries.json' and data_queries), \
                     patch.object(Path, 'exists', lambda p: str(p) == '/input' and input_exists), \
                     patch.object(Path, 'is_dir', lambda p: False):
                    result = resolve([], {})
                self.assertEqual(result.input, Path(expected))
                self.assertEqual(result.output, Path('/output/submission.csv'))
                self.assertEqual(result.argv[4:], ['--profile', 'contest', '--mode', 'llm'])

    def test_flags_pass_through(self):
        result = resolve(['--workers', '3', 'data', '--profile=dev-glm', '--mode', 'dump',
                          '--set', 'budget=0.01', '--set=stream=false', '--fresh',
                          '--checkpoint', 'resume.jsonl', '--config', 'custom.json',
                          '--dump-dir', 'prompts', '--no-checkpoint', '--self-consistency'], {})
        self.assertEqual(result.input, self.root/'data')
        for flag, value in [('profile', 'dev-glm'), ('mode', 'dump'), ('workers', '3'),
                            ('checkpoint', 'resume.jsonl'), ('config', str(self.root/'custom.json')),
                            ('dump-dir', 'prompts')]:
            self.assertEqual(result.argv[result.argv.index('--' + flag) + 1], value)
        self.assertEqual(result.argv.count('--profile'), 1)
        self.assertEqual(result.argv.count('--mode'), 1)
        self.assertEqual(result.argv.count('--set'), 2)
        self.assertTrue(all(flag in result.argv for flag in ('--fresh', '--no-checkpoint', '--self-consistency')))

    def test_key_aliases(self):
        for alias in ('OPENROUTER_KEY', 'OPENROUTER_API', 'API_KEY'):
            with self.subTest(alias=alias):
                env = {alias: 'test-secret'}
                result = resolve([], env)
                self.assertEqual(result.env['OPENROUTER_API_KEY'], 'test-secret')
                self.assertEqual(env, {alias: 'test-secret'})
                self.assertNotIn('test-secret', result.argv)

    def test_key_precedence_and_explicit_empty(self):
        aliases = {'OPENROUTER_KEY': 'first', 'OPENROUTER_API': 'second', 'API_KEY': 'third'}
        self.assertEqual(resolve([], aliases).env['OPENROUTER_API_KEY'], 'first')
        for key in ('canonical', ''):
            self.assertEqual(resolve([], dict(aliases, OPENROUTER_API_KEY=key)).env['OPENROUTER_API_KEY'], key)

    def test_no_key_run_from_scratch(self):
        data = self.root/'data'
        data.mkdir()
        (data/'queries.json').write_text(json.dumps([
            {'id': 'q1', 'file_name': 'missing.xlsx', 'query': 'How many?'}]))
        output = self.root/'new-output'
        result = subprocess.run([sys.executable, str(ROOT/'docker-entrypoint.py'), str(data), str(output)],
                                cwd=self.root, env=dict(os.environ, OPENROUTER_API_KEY=''),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output/'submission.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ['id', 'answer'])
            self.assertEqual(list(reader), [{'id': 'q1', 'answer': '0'}])
        self.assertIn('Input: ' + str(data), result.stdout)
        self.assertFalse((self.root/'runs').exists())

    def test_output_permissions_after_main(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                output = self.root / str(existing) / 'submission.csv'
                if existing:
                    output.parent.mkdir(mode=0o700)

                def run(*args, **kwargs):
                    output.write_text('id,answer\nq1,0\n')
                    output.chmod(0o600)
                    return subprocess.CompletedProcess(args[0], 0)

                previous = os.umask(0o077)
                try:
                    with patch.object(entrypoint.subprocess, 'run', run):
                        result = entrypoint.main(['data', str(output)])
                finally:
                    os.umask(previous)

                self.assertEqual(result, 0)
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)
                self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode),
                                 0o700 if existing else 0o755)

    def test_output_chmod_is_best_effort(self):
        output = self.root / 'output' / 'submission.csv'
        with patch.object(entrypoint.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)), \
             patch.object(Path, 'chmod', side_effect=PermissionError('not owned')):
            self.assertEqual(entrypoint.main(['data', str(output)]), 7)


if __name__ == '__main__':
    unittest.main()
