"""Tests of homework 1: forward-mode automatic differentiation.

The points x are drawn at import from a random generator seeded by the environment
variable HW01_SEED (default 0), so the tests can be repeated at other points. Run
`uv run pytest -x` and work from R1 to R7. Where a function of the module is composed
with an inner function (t * t, t / 2), the inner derivative differs from 1, so a rule
that forgets the chain-rule factor `x.deriv` does not pass.
"""

import math
import os
import random

import pytest
from dual import Dual, cos, derivative, exp, gradient, lift, log, sin, sqrt, tan

SEED = int(os.environ.get("HW01_SEED", "0"))
rng = random.Random(SEED)
POINTS = [
    round(rng.uniform(0.5, 1.4), 6) for _ in range(4)
]  # below the pole of tan at pi/2
H = 1e-5  # step of the central difference used as the oracle (the optimal step of ex02)


def central(f, x, h=H):
    return (f(x + h) - f(x - h)) / (2 * h)


def close(value, expected):
    return value == pytest.approx(expected, rel=1e-6, abs=1e-8)


# --- R1: arithmetic, as in ex04 -------------------------------------------------------


def test_r1_arithmetic_exact():
    x = Dual(3.0, 1.0)
    assert x + 1 == Dual(4.0, 1.0)
    assert 1 + x == Dual(4.0, 1.0)
    assert x - 1 == Dual(2.0, 1.0)
    assert 1 - x == Dual(-2.0, -1.0)
    assert 2 * x == Dual(6.0, 2.0)
    assert x * x == Dual(9.0, 6.0)
    assert -x == Dual(-3.0, -1.0)


def test_r1_derivative_of_polynomial_exact():
    assert derivative(lambda x: x * x * x - 2 * x + 1)(2.0) == 10.0


@pytest.mark.parametrize("x", POINTS)
def test_r1_product_rule(x):
    f = lambda t: (t + 2) * (t * t - 1)
    assert close(derivative(f)(x), central(f, x))


# --- R2: quotient -------------------------------------------------------------------------


@pytest.mark.parametrize("x", POINTS)
def test_r2_quotient_rule(x):
    f = lambda t: (t + 1) / (t * t + 1)
    assert close(derivative(f)(x), central(f, x))


def test_r2_reciprocal():
    y = 1 / Dual(2.0, 1.0)
    assert y.value == 0.5
    assert y.deriv == pytest.approx(-0.25)
    assert derivative(lambda x: 1 / x)(2.0) == pytest.approx(-0.25)


def test_r2_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        Dual(1.0, 1.0) / Dual(0.0, 1.0)
    with pytest.raises(ZeroDivisionError):
        Dual(1.0, 1.0) / 0


# --- R3: powers ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, -1, -2])
def test_r3_integer_power(n):
    x = POINTS[0]
    assert (Dual(x, 1.0) ** n).value == pytest.approx(x**n, rel=1e-12)
    assert close(derivative(lambda t: t**n)(x), 0.0 if n == 0 else n * x ** (n - 1))
    assert close(derivative(lambda t: (t + 1) ** n)(x), 0.0 if n == 0 else n * (x + 1) ** (n - 1))


def test_r3_zero_base():
    assert Dual(0.0, 1.0) ** 0 == Dual(1.0, 0.0)
    assert Dual(0.0, 1.0) ** 2 == Dual(0.0, 0.0)
    assert derivative(lambda t: t**3)(0.0) == 0.0


def test_r3_integer_power_negative_base():
    assert derivative(lambda t: t**3)(-2.0) == pytest.approx(12.0)
    assert (Dual(-2.0, 1.0) ** 2) == Dual(4.0, -4.0)


@pytest.mark.parametrize("p", [0.5, 1.5, -0.5, 2.5])
def test_r3_real_power(p):
    x = POINTS[1]
    assert (Dual(x) ** p).value == pytest.approx(x**p)
    assert close(derivative(lambda t: t**p)(x), p * x ** (p - 1))
    assert close(derivative(lambda t: (t * t) ** p)(x), p * (x * x) ** (p - 1) * 2 * x)


def test_r3_real_power_needs_positive_base():
    with pytest.raises(ValueError):
        Dual(-2.0, 1.0) ** 0.5


@pytest.mark.parametrize("c", [2.0, 10.0, math.e])
def test_r3_number_to_a_dual_power(c):
    x = POINTS[2]
    assert close(derivative(lambda t: c**t)(x), c**x * math.log(c))
    assert close(derivative(lambda t: c ** (t * t))(x), c ** (x * x) * math.log(c) * 2 * x)


# --- R4: elementary functions, each for a float and for a Dual ----------------------------

FUNCTIONS = [
    (sin, math.sin, math.cos),
    (cos, math.cos, lambda t: -math.sin(t)),
    (tan, math.tan, lambda t: 1 / math.cos(t) ** 2),
    (exp, math.exp, math.exp),
    (log, math.log, lambda t: 1 / t),
    (sqrt, math.sqrt, lambda t: 0.5 / math.sqrt(t)),
]


@pytest.mark.parametrize("x", POINTS)
@pytest.mark.parametrize(
    "f, ref, dref", FUNCTIONS, ids=[f[0].__name__ for f in FUNCTIONS]
)
def test_r4_value_and_derivative(f, ref, dref, x):
    assert isinstance(f(x), float)  # a plain number in, a plain number out
    assert f(x) == pytest.approx(ref(x), rel=1e-12)
    assert close(derivative(f)(x), dref(x))


@pytest.mark.parametrize("x", POINTS)
@pytest.mark.parametrize(
    "f, ref, dref", FUNCTIONS, ids=[f[0].__name__ for f in FUNCTIONS]
)
def test_r4_chain_rule(f, ref, dref, x):
    g = lambda t: f(t * t)  # inner derivative 2t: f(t*t)' = f'(t*t) * 2t
    assert close(derivative(g)(x), dref(x * x) * 2 * x)


def test_r4_domain_errors_as_for_floats():
    with pytest.raises(ValueError):
        log(Dual(-1.0, 1.0))
    with pytest.raises(ValueError):
        log(-1.0)
    with pytest.raises(ZeroDivisionError):
        sqrt(Dual(0.0, 1.0))


# --- R5: lift accepts numbers only ------------------------------------------------------


def test_r5_lift_numbers():
    assert lift(2) == Dual(2.0, 0.0)
    assert lift(2.5) == Dual(2.5, 0.0)
    d = Dual(1.0, 2.0)
    assert lift(d) is d


@pytest.mark.parametrize("bad", ["3", None, [1.0]], ids=["str", "None", "list"])
def test_r5_lift_rejects_non_numbers(bad):
    with pytest.raises(TypeError):
        lift(bad)
    with pytest.raises(TypeError):
        Dual(1.0) + bad


# --- R6: composite functions against the central difference ------------------------------

COMPOSITES = {
    "sin*exp": lambda t: sin(t) * exp(-t),
    "log(t^2)/t": lambda t: log(t * t) / t,
    "sqrt(1+t^2)": lambda t: sqrt(1 + t * t),
    "(1+t)^2.5/(1+t^2)": lambda t: (1 + t) ** 2.5 / (1 + t * t),
    "tan(t/2)/(1+exp)": lambda t: tan(t / 2) / (1 + exp(t)),
    "2^(t^2)*cos(2t)-1/(t+1)": lambda t: 2 ** (t * t) * cos(2 * t) - 1 / (t + 1),
}


@pytest.mark.parametrize("x", POINTS)
@pytest.mark.parametrize("g", COMPOSITES.values(), ids=COMPOSITES.keys())
def test_r6_composite_matches_central_difference(g, x):
    assert close(derivative(g)(x), central(g, x))


# --- R7: partial derivatives ----------------------------------------------------------------


def test_r7_gradient_two_variables():
    f = lambda x, y: x * x * y + sin(y)
    x0, y0 = POINTS[0], POINTS[1]
    gx, gy = gradient(f, [x0, y0])
    assert close(gx, 2 * x0 * y0)
    assert close(gy, x0 * x0 + math.cos(y0))


def test_r7_gradient_three_variables_vs_central_difference():
    f = lambda x, y, z: exp(x * y) / (1 + z * z) + log(x + y + z)
    point = POINTS[:3]
    grads = gradient(f, point)
    assert len(grads) == 3
    for i in range(3):

        def along(t, i=i):
            args = list(point)
            args[i] = t
            return f(*args)

        assert close(grads[i], central(along, point[i]))


def test_r6_function_returning_a_plain_number():
    assert derivative(lambda t: 2.0)(1.0) == 0.0


def test_r7_gradient_of_a_constant():
    assert gradient(lambda x, y: 1.0, [1.0, 2.0]) == [0.0, 0.0]