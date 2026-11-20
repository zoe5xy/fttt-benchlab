"""The standalone package must not import any of the original projects."""

import subprocess
import sys
from tests.helpers import source_environment


def test_independent_import():
    program = """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('usctbench', 'eitbench', 'inv_framework', 'fttt_framework'):
        raise AssertionError('unexpected original-project import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import ftttbench as library
assert library.describe()['algorithm_id'] == 'dip'
"""
    subprocess.run(
        [sys.executable, "-c", program], check=True, env=source_environment()
    )


def test_import_without_unix_resource_module():
    program = """
import builtins
original = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError('resource is unavailable', name='resource')
    return original(name, *args, **kwargs)
builtins.__import__ = without_resource
import ftttbench
import ftttbench.benchmark.runner as runner
assert runner.resource is None
assert [entry.name for entry in ftttbench.list_algorithms()] == ['dip', 'inr', 'dps']
"""
    subprocess.run(
        [sys.executable, "-c", program], check=True, env=source_environment()
    )
