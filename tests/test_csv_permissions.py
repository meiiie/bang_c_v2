import os
import stat

import pytest

from solver.config import write_csv


@pytest.mark.parametrize('umask', [0o022, 0o077])
def test_write_csv_is_world_readable(tmp_path, umask):
    output = tmp_path / 'output' / 'submission.csv'
    previous = os.umask(umask)
    try:
        write_csv(output, [{'id': 'q1', 'answer': '42'}])
    finally:
        os.umask(previous)

    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o755
    assert output.read_text() == 'id,answer\nq1,42\n'


def test_write_csv_tolerates_parent_chmod_permission_error(tmp_path, monkeypatch):
    output = tmp_path / 'submission.csv'
    original_chmod = os.chmod

    def chmod(path, mode, *args, **kwargs):
        if path == output.parent:
            raise PermissionError('directory owned by another user')
        return original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, 'chmod', chmod)
    write_csv(output, [{'id': 'q1', 'answer': '42'}])

    assert stat.S_IMODE(output.stat().st_mode) == 0o644
