# %% [markdown]
# ### [ex02] Derivatives by finite differences: truncation error and rounding error

# %%
import math
from collections.abc import Callable

Function = Callable[[float], float]


def forward_difference(f: Function, x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def central_difference(f: Function, x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


x0 = 1.0
exact = math.cos(x0)  # (sin x)' = cos x

print(f"{'h':>7}  {'forward error':>14}  {'central error':>14}")
for k in range(1, 17):
    h = 10.0 ** -k
    e_fwd = abs(forward_difference(math.sin, x0, h) - exact)
    e_ctr = abs(central_difference(math.sin, x0, h) - exact)
    print(f"{h:7.0e}  {e_fwd:14.2e}  {e_ctr:14.2e}")

# %%
import matplotlib.pyplot as plt

hs = [10.0 ** -k for k in range(1, 17)]
errors_fwd = [abs(forward_difference(math.sin, x0, h) - exact) for h in hs]
errors_ctr = [abs(central_difference(math.sin, x0, h) - exact) for h in hs]

plt.loglog(hs, errors_fwd, "o-", label="forward")
plt.loglog(hs, errors_ctr, "s-", label="central")
plt.loglog(hs, hs, ":", color="grey", label="h")
plt.loglog(hs, [h * h for h in hs], "--", color="grey", label="h²")
plt.loglog(hs, [2.2e-16 / h for h in hs], "-.", color="grey", label="ε / h")
plt.xlabel("h")
plt.ylabel("absolute error of f'(1), f = sin")
plt.legend()
plt.show()


# %%
def observed_order(method, f: Function, x: float, exact: float, h1: float, h2: float) -> float:
    """The observed order p: error(h) ~ C * h**p, so p = log(e1 / e2) / log(h1 / h2)."""
    e1 = abs(method(f, x, h1) - exact)
    e2 = abs(method(f, x, h2) - exact)
    return math.log(e1 / e2) / math.log(h1 / h2)


print(observed_order(forward_difference, math.sin, x0, exact, 1e-2, 1e-3))
print(observed_order(central_difference, math.sin, x0, exact, 1e-2, 1e-3))


# %%
def derivative(f: Function, h: float = 1e-5) -> Function:
    """Return the function x -> f'(x), approximated by the central difference."""
    raise NotImplementedError  # TODO


# cos_approx = derivative(math.sin)
# print(cos_approx(0.0), cos_approx(math.pi))


def second_difference(f: Function, x: float, h: float) -> float:
    """f''(x) ~ (f(x + h) - 2 f(x) + f(x - h)) / h**2"""
    raise NotImplementedError  # TODO


# %%
def taylor_exp(x: float, n: int) -> float:
    """The Taylor polynomial of exp at 0: the sum of x**k / k! for k = 0..n."""
    total = 0.0
    term = 1.0  # x**0 / 0!
    for k in range(1, n + 1):
        total += term
        term *= x / k  # x**k / k! from x**(k - 1) / (k - 1)!
    return total + term


for n in [1, 2, 4, 8, 16]:
    print(n, taylor_exp(1.0, n), abs(taylor_exp(1.0, n) - math.e))

# %%
# truncation error: how many terms does x = 10 need?
for n in [10, 20, 30, 40]:
    print(n, taylor_exp(10.0, n), math.exp(10.0))

# rounding error: the same series at x = -20
print(taylor_exp(-20.0, 100), math.exp(-20.0))
print(1 / taylor_exp(20.0, 100), math.exp(-20.0))
