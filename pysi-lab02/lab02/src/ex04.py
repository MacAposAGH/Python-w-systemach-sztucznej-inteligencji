# %% [markdown]
# ### [ex04] Forward-mode automatic differentiation: `Dual` carries (value, derivative) pairs

# %%
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Dual:
    """A (value, derivative) pair: `value` is the value of an intermediate result, `deriv` its
    derivative; the same pair is PyTorch's dual tensor (primal, tangent) in forward_ad."""

    value: float
    deriv: float = 0.0

    def __add__(self, other) -> "Dual":
        other = lift(other)
        return Dual(self.value + other.value, self.deriv + other.deriv)

    __radd__ = __add__

    def __mul__(self, other) -> "Dual":
        other = lift(other)
        # (a, b) * (c, d) = (a c, a d + b c): the product rule (u v)' = u v' + u' v
        return Dual(self.value * other.value, self.value * other.deriv + self.deriv * other.value)

    __rmul__ = __mul__

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.deriv)

    def __sub__(self, other) -> "Dual":
        return self + (-lift(other))

    def __rsub__(self, other) -> "Dual":
        return lift(other) - self

    def __truediv__(self, other) -> "Dual":
        """(a, b) / (c, d) = (a / c, (b c - a d) / c**2): the quotient rule."""
        other = lift(other)
        Dual(self.value / other.value, (self.deriv * other.value - self.value * other.deriv) / (other.value ** 2))
        raise NotImplementedError  # TODO

    def __rtruediv__(self, other) -> "Dual":
        return lift(other) / self

    def __pow__(self, n: int) -> "Dual":
        """(a, b) ** n = (a**n, n a**(n - 1) b) for an integer n: the power rule."""
        raise NotImplementedError  # TODO


def lift(x) -> Dual:
    """A plain number is a constant: the pair (x, 0)."""
    return x if isinstance(x, Dual) else Dual(float(x))


x = Dual(3.0, 1.0)  # derivative 1: dx/dx = 1
print(x + 1, 2 * x, x * x)
print((x + 2) * (x + 1))  # f(x) = (x + 2)(x + 1) = x**2 + 3x + 2, f'(3) = 2 * 3 + 3 = 9


# %%
def derivative(f):
    """Return x -> f'(x): evaluate f at the pair (x, 1), since dx/dx = 1, and read `deriv`."""
    return lambda x: f(Dual(x, 1.0)).deriv


def poly(x):
    return x * x * x - 2 * x + 1  # f'(x) = 3 x**2 - 2


print(derivative(poly)(2.0), 3 * 2.0 ** 2 - 2)


# %%
# the chain rule: f(Dual(a, b)) = Dual(f(a), f'(a) * b); b is the derivative of the inner function
def sin(x):
    if not isinstance(x, Dual):
        return math.sin(x)
    return Dual(math.sin(x.value), math.cos(x.value) * x.deriv)


def cos(x):
    if not isinstance(x, Dual):
        return math.cos(x)
    return Dual(math.cos(x.value), -math.sin(x.value) * x.deriv)


def exp(x):
    if not isinstance(x, Dual):
        return math.exp(x)
    return Dual(math.exp(x.value), math.exp(x.value) * x.deriv)


def log(x):
    raise NotImplementedError  # TODO


def g(x):
    return sin(x) * exp(-x)  # g'(x) = (cos x - sin x) exp(-x)


x0 = 1.5
print(derivative(g)(x0), (math.cos(x0) - math.sin(x0)) * math.exp(-x0))


# %%
# three ways to a derivative: analytic (symbolic), numerical, automatic (forward mode)
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


exact = (math.cos(x0) - math.sin(x0)) * math.exp(-x0)
print(f"analytic           {exact:.17f}")
print(f"central, h = 1e-5  {central_difference(g, x0, 1e-5):.17f}")
print(f"forward-mode AD    {derivative(g)(x0):.17f}")
print(abs(central_difference(g, x0, 1e-5) - exact), abs(derivative(g)(x0) - exact))


# %%
def newton(f, x0: float, steps: int = 6) -> float:
    """Newton's method x <- x - f(x) / f'(x), with f' from `derivative`."""
    df = derivative(f)
    x = x0
    for _ in range(steps):
        x = x - f(x) / df(x)
        print(x)
    return x


print(newton(lambda x: x * x - 2, 1.0), math.sqrt(2))
