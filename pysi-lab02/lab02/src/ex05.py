# %% [markdown]
# ### [ex05] Exceptions and context managers

# %%
import math


def safe_log(x: float) -> float:
    try:
        return math.log(x)
    except ValueError as err:  # math domain error
        print(f"ValueError: {err}")
        return math.nan


print(safe_log(math.e), safe_log(-1.0), safe_log(0.0))


# %%
def parse_number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        print(f"not a number: {text!r}")
        raise  # the same exception continues to the caller
    else:
        print("parsed", value)  # only when nothing was raised
        return value
    finally:
        print(f"done with {text!r}")  # always: after return and after raise


parse_number("3.14")
try:
    parse_number("abc")
except ValueError:
    print("caught again, outside")

# %%
# the hierarchy: except catches the class and its subclasses
print(issubclass(ZeroDivisionError, ArithmeticError), issubclass(OverflowError, ArithmeticError))
for expr in ["1 / 0", "math.exp(1000)", "int('x')", "[1, 2][5]", "{}['k']"]:
    try:
        eval(expr)
    except ArithmeticError as err:
        print(f"{expr:16} -> ArithmeticError: {type(err).__name__}")
    except (LookupError, ValueError) as err:
        print(f"{expr:16} -> {type(err).__name__}: {err}")


# %%
class DomainError(ValueError):
    """Raised when a function is evaluated outside its domain."""


def sqrt_checked(x: float) -> float:
    if x < 0:
        raise DomainError(f"square root of a negative number: {x}")
    return math.sqrt(x)


try:
    sqrt_checked(-4.0)
except DomainError as err:
    print(type(err).__name__, err)


def mean(xs: list[float]) -> float:
    try:
        return sum(xs) / len(xs)
    except ZeroDivisionError as err:
        raise ValueError("mean of an empty list") from err  # the cause stays attached


try:
    mean([])
except ValueError as err:
    print(err, "| caused by:", repr(err.__cause__))

# %%
# a file is a resource: `with` closes it, also when the block raises
hs = [10.0 ** -k for k in range(1, 9)]
with open("errors.csv", "w", encoding="utf-8") as out:
    out.write("h,error\n")
    for h in hs:
        err = abs((math.sin(1 + h) - math.sin(1 - h)) / (2 * h) - math.cos(1))
        out.write(f"{h:.0e},{err:.3e}\n")
print(out.closed)

with open("errors.csv", encoding="utf-8") as src:
    for line in src:
        print(line.rstrip())

# %%
import time


class Timer:
    """with Timer() as t: ...   then t.elapsed holds the seconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.elapsed = time.perf_counter() - self.start
        status = f" (after {exc_type.__name__})" if exc_type else ""
        print(f"elapsed: {self.elapsed:.4f} s{status}")
        return False  # False: the exception, if any, continues


with Timer() as t:
    total = sum(i * i for i in range(1_000_000))
print(t.elapsed)

try:
    with Timer():
        1 / 0
except ZeroDivisionError:
    print("the exception passed through __exit__")

# %%
import os
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield  # paused while the with block runs in the caller
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f} s")


with timer("sum of squares"):
    total = sum(i * i for i in range(1_000_000))


@contextmanager
def working_directory(path: str):
    """Run the block inside `path`, then come back, also after an exception."""
    raise NotImplementedError  # TODO


# with working_directory(".."):
#     print(os.getcwd())
# print(os.getcwd())
