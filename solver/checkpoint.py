"""Durable successful answers, matched against question and workbook content."""
import hashlib
import json
import os
from pathlib import Path


def answer_config_hash(config):
    # Execution placement, credentials and checkpoint location do not change answers.
    ignored = {'checkpoint_path', 'workers', 'max_abandoned_workers', 'program_workers', 'profiles', 'profile',
               'api_key_env', 'total_time_limit', 'time_guard_margin'}
    settings = {k: v for k, v in config.items() if k not in ignored}
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def answer_key(query, workbook_hash, model, config_hash):
    parts = [str(query['id']), workbook_hash, query['query'], model, config_hash]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class Checkpoint:
    """Caller serializes appends; malformed records never become cache hits."""
    def __init__(self, path, fresh=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.answers = {}
        if fresh:
            with self.path.open('wb') as stream:
                stream.flush()
                os.fsync(stream.fileno())
        elif self.path.exists():
            with self.path.open('rb') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if (isinstance(row, dict) and isinstance(row.get('key'), str)
                                and isinstance(row.get('id'), str)
                                and isinstance(row.get('answer'), str) and row['answer']
                                and not row.get('error') and not row.get('fallback')):
                            self.answers[row['key']] = row
                    except (ValueError, UnicodeError):
                        continue
            # Separate future appends from an interrupted final write, including
            # valid JSON whose newline was lost. Do not discard earlier records.
            with self.path.open('rb+') as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell():
                    stream.seek(-1, os.SEEK_END)
                    if stream.read(1) != b'\n':
                        stream.write(b'\n')
                        stream.flush()
                        os.fsync(stream.fileno())

    def append(self, key, row):
        record = {'key': key, 'id': str(row['id']), 'answer': row['answer']}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.answers[key] = record
