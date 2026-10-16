from pathlib import Path
import pytest


@pytest.fixture(scope='session')
def data_dir():
    path = Path(__file__).resolve().parents[2]/'data'
    if not path.exists(): pytest.skip('Development workbooks not installed')
    return path


@pytest.fixture(scope='session')
def oracle_programs():
    path = Path(__file__).parent / 'oracle_programs'
    if not path.is_dir():
        pytest.skip('Optional oracle programs not installed')
    return path
