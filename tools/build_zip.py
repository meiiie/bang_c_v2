#!/usr/bin/env python3
"""Build the runtime archive from required entry files and allowlisted runtime paths."""
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'README.md', 'main.py', 'eval.py', 'requirements.txt', 'configs/default.json',
            'Dockerfile', '.dockerignore', 'docker-entrypoint.py', 'LICENSE'}


def build(root=ROOT, destination=None):
    root = Path(root)
    destination = Path(destination) if destination else root/'dist/submission.zip'
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    # A fresh export may have no tracked runtime files yet. Restrict the fallback
    # to source/config extensions; never sweep input workbooks or answer files.
    runtime = {name for name in tracked if name.startswith(('solver/', 'configs/'))}
    if not runtime:
        runtime = {str(path.relative_to(root))
                   for folder, pattern in (('solver', '*.py'), ('configs', '*.json'))
                   for path in (root/folder).rglob(pattern)}
    # Required entry files may be new working-tree additions before a commit.
    files = sorted(REQUIRED | {name for name in runtime
                   if name.startswith(('solver/', 'configs/')) and
                   '__pycache__' not in Path(name).parts and not name.endswith('.pyc')})
    missing = {name for name in REQUIRED if not (root/name).is_file()}
    if missing:
        raise RuntimeError(f'Required files are missing: {sorted(missing)}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = root/name
            if path.is_symlink():
                raise ValueError(f'Archive refuses symlink: {name}')
            archive.write(path, name)
    size = destination.stat().st_size
    assert size < 100 * 1024**2, f'Archive too large: {size} bytes'
    print(f'{destination}: {size:,} bytes ({size/1024**2:.3f} MiB), {len(files)} files')
    return destination


if __name__ == '__main__':
    build()
