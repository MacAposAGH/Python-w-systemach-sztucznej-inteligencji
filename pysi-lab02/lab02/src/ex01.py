# %% [markdown]
# ### [ex01] Functions as values, closures and decorators

# %%
import math
import time
from collections.abc import Callable
from functools import wraps, lru_cache

# %%

# a function is an object: it can be passed, stored and returned like any other value
print(type(math.sqrt), math.sqrt.__name__)


def apply_twice(f: Callable[[float], float], x: float) -> float:
    return f(f(x))


print(apply_twice(math.sqrt, 16.0))
print(apply_twice(lambda x: 2 * x + 1, 1.0))

fs = [math.sin, math.cos, math.exp, lambda x: x ** 2]
print([f(0.0) for f in fs])
print([f.__name__ for f in fs])

l = lambda x: x ** 2


# %%
def make_power(n: int) -> Callable[[float], float]:
    """Return the function x -> x ** n."""

    def power(x: float) -> float:
        return x ** n  # n is a free variable: it lives in the enclosing scope

    return power


square = make_power(2)
cube = make_power(3)
print(square(3.0), cube(3.0))
print(square.__closure__[0].cell_contents)  # the value captured by the closure

# late binding
fs = [lambda x: x ** n for n in range(3)]
for i, f in enumerate(fs):
    print(f(i + 1))


# %%
def make_counter() -> Callable[[], int]:
    count = 0

    def counter() -> int:
        nonlocal count  # without it, `count += 1` would create a new local variable
        # count = 0
        count += 1
        return count

    return counter


c1 = make_counter()
c2 = make_counter()
print(c1(), c1(), c1(), c2())  # each closure has its own count


# mine
def normal_count() -> int:
    count = 0
    return count + 1


print(normal_count(), normal_count())


# %%

# mine - docstring example
def check_doc():
    """Check the docstring of the function."""
    print("Hello world")


print(check_doc.__doc__)


def timed(f: Callable) -> Callable:
    """A decorator: takes a function, returns a function that measures every call."""

    def wrapper(*args, **kwargs):
        """Mine:    General inner function docstring. Without it would be: 'None'
                    To use original function docstring use: g = timed(g) (See example below)"""
        start = time.perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__}: {time.perf_counter() - start:.4f} s")
        return result

    return wrapper


def sum_to(n: int) -> int:
    """Mine: Simple sum to function"""
    return sum(range(n))


sum_to = timed(sum_to)  # the name now refers to the wrapper
print(sum_to(10_000_000))

# mine
print(sum_to)
print(sum_to.__closure__[0].cell_contents)
print(sum_to.__closure__[0].cell_contents.__doc__)


@timed  # the same as: sum_squares = timed(sum_squares)
def sum_squares(n: int) -> int:
    """Sum of i * i for i < n."""
    return sum(i * i for i in range(n))


print(sum_squares(1_000_000))
print(sum_squares.__name__, sum_squares.__doc__)  # the wrapper hides the original


# %%

def repeat(times: int) -> Callable:
    """A decorator with a parameter: repeat(3) returns the actual decorator."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)  # copies __name__, __doc__, ... from f to wrapper
        def wrapper(*args, **kwargs):
            result = None
            for _ in range(times):
                result = f(*args, **kwargs)
            return result

        return wrapper

    return decorator


@repeat(3)
def greet(name: str) -> str:
    """Say hello."""
    print(f"hello, {name}")
    return name


print(greet("Ada"))
print(greet.__name__, greet.__doc__)

# %%
from itertools import islice


def taylor_terms(x):
    term, k = 1.0, 0
    while True:
        yield term
        k += 1
        term *= x / k
        # print(term)


print(list(islice(taylor_terms(1), 4)))

# %%

n = 400

def fib(n: int) -> int:
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(timed(fib)(30))

def fib_clarified(n: int) -> int:
    if n < 2:
        return n
    a = fib_clarified(n - 1)
    b = fib_clarified(n - 2)
    return a + b

list = [0, 1]
def fib_memoization_v1(n: int) -> int:
    if n < len(list):
        return list[n]
    list.append(fib_memoization_v1(n - 1) + fib_memoization_v1(n - 2))
    return list[n]

f0 = 0
def fib_memoization_v2(n: int) -> int:
    global f0
    if n < 2:
        return n
    result = fib_memoization_v2(n - 1) + f0
    f0 = result - f0
    return result

print(timed(fib_memoization_v2)(n))

def fib_memoization_v2_decorator() -> Callable:
    f_0 = 0
    def fib_memoization_v2_wrapper(n: int) -> int:
        nonlocal f_0
        if n < 2:
            return n
        result = fib_memoization_v2_wrapper(n - 1) + f_0
        f_0 = result - f_0
        return result

    return fib_memoization_v2_wrapper

print(timed(fib_memoization_v2_decorator())(n))

def cached() -> Callable:
    """A decorator that remembers the result of every call (memoization)."""

    memoization = {0: 0, 1: 1}
    def wrapper(n: int) -> int:
        value = memoization.get(n)
        if value:
            return value
        memoization[n] = wrapper(n - 1) + memoization[n - 2]
        return memoization[n]

    return wrapper

print(timed(cached())(n))

# the standard library has it: functools.cache (and lru_cache with a size limit)
from functools import cache

@cache
# @lru_cache
def fib_cached(n: int) -> int:
    return n if n < 2 else fib_cached(n - 1) + fib_cached(n - 2)

print(timed(fib_cached)(n))
print(fib_cached.cache_info())
