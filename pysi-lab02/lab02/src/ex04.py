# %% [markdown]
# ### [ex04] Dual numbers: exact derivatives by forward-mode automatic differentiation

# %%
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dual:
    """a + b*eps with eps**2 = 0: `real` is the value, `dual` carries the derivative."""

    real: float
    dual: float = 0.0

    def __add__(self, other) -> "Dual":
        other = lift(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    __radd__ = __add__

    def __mul__(self, other) -> "Dual":
        other = lift(other)
        # (a + b eps)(c + d eps) = ac + (ad + bc) eps, because eps**2 = 0: the product rule
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)

    __rmul__ = __mul__

    def __neg__(self) -> "Dual":
        return Dual(-self.real, -self.dual)

    def __sub__(self, other) -> "Dual":
        return self + (-lift(other))

    def __rsub__(self, other) -> "Dual":
        return lift(other) - self

    def __truediv__(self, other) -> "Dual":
        """(a + b eps) / (c + d eps): the quotient rule."""
        raise NotImplementedError  # TODO

    def __rtruediv__(self, other) -> "Dual":
        return lift(other) / self

    def __pow__(self, n: int) -> "Dual":
        """(a + b eps) ** n for an integer n: the power rule."""
        raise NotImplementedError  # TODO


def lift(x) -> Dual:
    """A plain number is a dual number with no derivative part."""
    return x if isinstance(x, Dual) else Dual(float(x))


x = Dual(3.0, 1.0)  # dual part 1: x changes at rate 1 with respect to itself
print(x + 1, 2 * x, x * x)
print((x + 2) * (x + 1))  # f(x) = (x + 2)(x + 1) = x**2 + 3x + 2, f'(3) = 2 * 3 + 3 = 9


# %%
def derivative(f):
    """Return the function x -> f'(x): evaluate f at x + 1 eps and read the dual part."""
    return lambda x: f(Dual(x, 1.0)).dual


def poly(x):
    return x * x * x - 2 * x + 1  # f'(x) = 3 x**2 - 2


print(derivative(poly)(2.0), 3 * 2.0 ** 2 - 2)


# %%
# the chain rule: f(a + b eps) = f(a) + f'(a) b eps
def sin(x):
    if not isinstance(x, Dual):
        return math.sin(x)
    return Dual(math.sin(x.real), math.cos(x.real) * x.dual)


def cos(x):
    if not isinstance(x, Dual):
        return math.cos(x)
    return Dual(math.cos(x.real), -math.sin(x.real) * x.dual)


def exp(x):
    if not isinstance(x, Dual):
        return math.exp(x)
    return Dual(math.exp(x.real), math.exp(x.real) * x.dual)


def log(x):
    raise NotImplementedError  # TODO


def g(x):
    return sin(x) * exp(-x)  # g'(x) = (cos x - sin x) exp(-x)


x0 = 1.5
print(derivative(g)(x0), (math.cos(x0) - math.sin(x0)) * math.exp(-x0))


# %%
# three ways to a derivative: the formula, a finite difference, a dual number
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


exact = (math.cos(x0) - math.sin(x0)) * math.exp(-x0)
print(f"formula            {exact:.17f}")
print(f"central, h = 1e-5  {central_difference(g, x0, 1e-5):.17f}")
print(f"dual number        {derivative(g)(x0):.17f}")
print(abs(central_difference(g, x0, 1e-5) - exact), abs(derivative(g)(x0) - exact))


# %%
def newton(f, x0: float, steps: int = 6) -> float:
    """Newton's method x <- x - f(x) / f'(x), with f' from dual numbers."""
    df = derivative(f)
    x = x0
    for _ in range(steps):
        x = x - f(x) / df(x)
        print(x)
    return x


print(newton(lambda x: x * x - 2, 1.0), math.sqrt(2))
