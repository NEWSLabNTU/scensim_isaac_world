"""Plain test functions (pytest's style) for unittest too, with no pytest (issue 090).

The suite runs under `unittest discover` (scensim's `just test`, and in Isaac's own venv, which
has no pytest), where a module of bare `test_` functions holds no tests at all, and one that
imports pytest does not import. `approx` compares as pytest.approx does by default (relative
1e-6, absolute 1e-12; `abs=` and `rel=` as there), scalars or arrays; `load_tests` makes each
`test_` function of a module a unittest case:

    from _plain import approx, cases
    load_tests = cases(__name__)
"""
import inspect
import sys
import unittest

import numpy as np


class approx:  # noqa: N801 - read as pytest's
    __array_ufunc__ = None   # numpy defers `array == approx(...)` to __eq__, as with pytest's
    __array_priority__ = 100
    def __init__(self, expected, rel=None, abs=None):  # noqa: A002 - pytest's argument names
        self.expected = expected
        self.rel = 1e-6 if rel is None and abs is None else (rel or 0.0)
        self.abs = 1e-12 if abs is None else abs

    def __eq__(self, actual):
        a, e = np.asarray(actual, dtype=float), np.asarray(self.expected, dtype=float)
        if a.shape != e.shape and a.size != 1 and e.size != 1:
            return False
        return bool(np.all(np.abs(a - e) <= np.maximum(self.abs, self.rel * np.abs(e))))

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


def cases(module_name):
    """A `load_tests` for the module `module_name`: its `test_` functions as cases."""
    def load(loader, tests, pattern):
        module = sys.modules[module_name]
        for name, f in sorted(vars(module).items()):
            if name.startswith("test_") and inspect.isfunction(f):
                tests.addTest(unittest.FunctionTestCase(f, description=f"{module_name}.{name}"))
        return tests
    return load
