# %% [markdown]
# ### [ex06] Iterators, generators and type annotations

# %%
import itertools
import math
import sys
from collections.abc import Iterable, Iterator


def taylor_terms(x: float) -> Iterator[float]:
    """x**k / k! for k = 0, 1, 2, ...: an infinite sequence, one term at a time."""
    term = 1.0
    k = 0
    while True:
        yield term  # pause here; resume at the next request
        k += 1
        term *= x / k


terms = taylor_terms(1.0)
print(type(terms))
print(next(terms), next(terms), next(terms))
print(list(itertools.islice(taylor_terms(1.0), 6)))

# %%
# the iterator protocol: iter() returns an iterator, next() returns the elements,
# StopIteration ends the sequence; a for loop does exactly this
it = iter([10, 20, 30])
print(type(it), next(it), next(it), next(it))
try:
    next(it)
except StopIteration:
    print("StopIteration: a for loop stops here")
print(iter(terms) is terms)  # a generator is its own iterator


class Countdown:
    """What a generator does by hand: __iter__ returns the iterator, __next__ the elements."""

    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1


print(list(Countdown(3)))
for i in Countdown(2):  # for calls iter() once, then next() until StopIteration
    print(i)

# %%
# partial sums: the Taylor polynomials of exp, one after another
sums = itertools.accumulate(taylor_terms(1.0))
for n, s in enumerate(itertools.islice(sums, 12)):
    print(n, s, abs(s - math.e))


# %%
def exp_series(x: float, eps: float = 1e-16) -> float:
    """Sum the terms while they are larger than eps."""
    significant = itertools.takewhile(lambda t: abs(t) > eps, taylor_terms(x))
    return sum(significant)


print(exp_series(1.0), math.exp(1.0))
print(exp_series(-20.0), math.exp(-20.0))  # the rounding problem of ex02 stays

# %%
squares_list = [i * i for i in range(1_000_000)]
squares_gen = (i * i for i in range(1_000_000))
print(sys.getsizeof(squares_list), sys.getsizeof(squares_gen))
print(sum(squares_gen))
print(sum(squares_gen))  # a generator is used up after one pass


# %%
def batches(xs: list, size: int) -> Iterator[list]:
    """Consecutive slices of xs of length size, the last one possibly shorter (mini-batches)."""
    raise NotImplementedError  # TODO


# for b in batches(list(range(10)), 4):
#     print(b)  # [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]


def running_mean(xs: Iterable[float]) -> Iterator[float]:
    """The mean of the elements seen so far, one value per element."""
    raise NotImplementedError  # TODO


# print(list(running_mean([2.0, 4.0, 6.0])))  # [2.0, 3.0, 4.0]

# %%
# type annotations: documentation checked by tools (mypy, pyright), not by the interpreter
from collections.abc import Callable

Function = Callable[[float], float]


def compose(f: Function, g: Function) -> Function:
    return lambda x: f(g(x))


h = compose(math.sqrt, abs)
print(h(-16.0))
print(compose("not a function", abs))  # no error: annotations are not checked at run time
print(compose.__annotations__)
