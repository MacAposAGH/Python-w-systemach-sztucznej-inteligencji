# %% [markdown]
# ### [ex07] Tests with pytest

# %%
import math

import pytest

try:
    from ex04 import Dual, derivative, exp, sin  # as a file: `pytest ex07.py` in src/lab02
except ImportError:
    pass  # in the notebook the names are already defined by ex04


def test_add_and_mul():
    x = Dual(3.0, 1.0)
    assert x + 1 == Dual(4.0, 1.0)
    assert 2 * x == Dual(6.0, 2.0)
    assert x * x == Dual(9.0, 6.0)


def test_derivative_of_polynomial():
    assert derivative(lambda x: x * x * x - 2 * x + 1)(2.0) == 10.0


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, -3.0])
def test_derivative_of_sin_is_cos(x):
    assert derivative(sin)(x) == pytest.approx(math.cos(x))


def test_product_and_chain_rule():
    x0 = 1.5
    expected = (math.cos(x0) - math.sin(x0)) * math.exp(-x0)
    assert derivative(lambda x: sin(x) * exp(-x))(x0) == pytest.approx(expected)


def test_division():
    assert derivative(lambda x: 1 / x)(2.0) == pytest.approx(-0.25)


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        Dual(1.0, 1.0) / Dual(0.0, 1.0)


# %%
def run_tests(namespace: dict) -> None:
    """A small stand-in for pytest inside a notebook: calls every test_* function."""
    for name, f in list(namespace.items()):
        if not name.startswith("test_") or not callable(f):
            continue
        marks = [m for m in getattr(f, "pytestmark", []) if m.name == "parametrize"]
        argsets = [v if isinstance(v, tuple) else (v,) for v in marks[0].args[1]] if marks else [()]
        for args in argsets:
            try:
                f(*args)
                print(f"PASS  {name}{args if args else ''}")
            except Exception as err:
                print(f"FAIL  {name}{args if args else ''}: {type(err).__name__}: {err}")


if __name__ == "__main__":  # true in the notebook and for `python ex07.py`, not under pytest
    run_tests(globals())
