"""Forward-mode automatic differentiation with the class Dual (homework 1).

A Dual is a (value, derivative) pair: `value` is the value of an intermediate result,
`deriv` its derivative (PyTorch: a dual tensor with primal and tangent). Every
operation below computes the `value` field of the result from the `value` fields of its
operands and the `deriv` field by a rule of differentiation; a function written for plain
numbers then computes its own derivative when called with a Dual.

Every function and method marked TODO raises NotImplementedError until completed;
the docstring of each gives the rule to implement. Work in the order of the tests
(`uv run pytest -x`): R1 arithmetic (as in ex04), R2 quotient, R3 powers, R4 elementary
functions, R5 lift, R6/R7 derivatives and gradient.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Number = (int, float)


@dataclass(frozen=True)
class Dual:
    """The pair (a, b): `value` is the value a, `deriv` the derivative b."""

    value: float
    deriv: float = 0.0

    # --- R1: sum, difference, product (ex04) ---------------------------------------
    def __add__(self, other) -> "Dual":
        other = lift(other)
        return Dual(self.value + other.value, self.deriv + other.deriv)

    __radd__ = __add__

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.deriv)

    def __sub__(self, other) -> "Dual":
        return self + (-lift(other))

    def __rsub__(self, other) -> "Dual":  # other - self
        return lift(other) - self

    def __mul__(self, other) -> "Dual":
        other = lift(other)
        return Dual(self.value * other.value, self.value * other.deriv + self.deriv * other.value)

    __rmul__ = __mul__

    # --- R2: quotient ----------------------------------------------------------------
    def __truediv__(self, other) -> "Dual":
        """(a, b) / (c, d) = (a / c, (b c - a d) / c**2); ZeroDivisionError for c = 0"""
        other = lift(other)
        return Dual(self.value / other.value, (self.deriv * other.value - self.value * other.deriv) / (other.value ** 2))

    def __rtruediv__(self, other) -> "Dual":  # other / self
        return lift(other) / self

    # --- R3: powers ------------------------------------------------------------------
    def __pow__(self, p) -> "Dual":
        """(a, b) ** p = (a**p, p a**(p-1) b) for a plain exponent p.

        An int exponent works for every base (p = 0 gives derivative 0); a non-integer
        exponent needs a > 0 (ValueError otherwise). A Dual exponent is not supported
        (TypeError).
        """
        if isinstance(p, Dual):
            raise TypeError(f"Expected int or float, got Dual")
        if isinstance(p, float) and self.value < 0:
            raise ValueError(f"Base must be positive for non-integer exponent")
        return Dual(self.value ** p, p * (self.value ** (p - 1)) * self.deriv)

    def __rpow__(self, c) -> "Dual":
        """c ** (a, b) = (c**a, c**a ln(c) b) for a plain base c > 0"""
        if c <= 0:
            raise ValueError(f"Base must be positive")
        return Dual(self.value ** c, c ** self.value * math.log(c) * self.deriv)


def lift(x) -> Dual:
    """A plain number as a constant, the pair (x, 0); a Dual unchanged.

    Anything else (a string, None, a list) is rejected with TypeError: a silent
    float("3") would turn a programming error into a wrong result.
    """
    if isinstance(x, (int, float)):
        return Dual(float(x), 0)
    if isinstance(x, Dual):
        return x
    raise TypeError(f"Expected Dual, int or float, got {type(x).__name__}")


# --- R4: elementary functions, each accepting a float and a Dual ----------------------
# f(Dual(a, b)) = Dual(f(a), f'(a) * b): the chain rule; `x.deriv` is the derivative of the inner function

def sin(x):
    x = lift(x)
    return Dual(math.sin(x.value), math.cos(x.value) * x.deriv)


def cos(x):
    x = lift(x)
    return Dual(math.cos(x.value), -math.sin(x.value) * x.deriv)


def tan(x):
    """tan' = 1 / cos**2"""
    x = lift(x)
    return Dual(math.tan(x.value), 1 / (math.cos(x.value) ** 2) * x.deriv)


def exp(x):
    x = lift(x)
    return Dual(math.exp(x.value), math.exp(x.value) * x.deriv)


def log(x):
    """log' = 1 / x; math.log raises ValueError for x <= 0, as for floats"""
    if isinstance(x, float) and x <= 0:
        raise ValueError(f"Base must be positive")
    x = lift(x)
    return Dual(math.log(x.value), (1 / x.value) * x.deriv)


def sqrt(x):
    """sqrt' = 1 / (2 sqrt x); x = 0 raises ZeroDivisionError (the derivative is infinite)"""
    if isinstance(x, float) and x == 0:
        raise ZeroDivisionError
    return Dual(x.value ** 2, (1 / (2 * math.sqrt(x.value))) * x.deriv)


# --- R6/R7: derivatives of user functions -------------------------------------------------

def derivative(f: Callable) -> Callable[[float], float]:
    """x -> f'(x): evaluate f at the pair (x, 1), since dx/dx = 1, and read `deriv`.

    `lift` makes a function that returns a plain number (a constant) work too.
    """

    def inner_derivative(x: float) -> float:
        if not isinstance(x, Dual):
            return f(x)
        return f(Dual(x, 1.0)).deriv
        # try:
        #     d = Dual(x, 1.0)
        #     f_res = f(d)
        #     return lift(f_res).deriv
        # except AttributeError:
        #     raise AttributeError

    return lambda x: f(Dual(x, 1.0)).deriv
    # return inner_derivative


def gradient(f: Callable, point: Sequence[float]) -> list[float]:
    """Partial derivatives of f(x_1, ..., x_n) at `point`: one forward pass per variable.

    In pass i the i-th argument is Dual(x_i, 1.0) and the others Dual(x_j, 0.0).
    """
    raise NotImplementedError  # TODO
