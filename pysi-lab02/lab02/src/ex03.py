# %% [markdown]
# ### [ex03] Classes, data classes and special methods: a vector in the plane

# %%
import math


class Vector2D:
    """A vector in the plane, written by hand."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Vector2D) and (self.x, self.y) == (other.x, other.y)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, k: float) -> "Vector2D":  # v * k
        return Vector2D(k * self.x, k * self.y)

    def __rmul__(self, k: float) -> "Vector2D":  # k * v
        return self * k

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)


u = Vector2D(3.0, 4.0)
v = Vector2D(1.0, 2.0)
print(u, u + v, 2 * u, u * 2, abs(u))
print(u == Vector2D(3.0, 4.0), u is Vector2D(3.0, 4.0))
print(u * v)  # no error, but what is it?


# %%
class Point:
    """The same data without the special methods."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


p = Point(1.0, 2.0)
print(p)
print(p == Point(1.0, 2.0))  # the default __eq__ compares identity
try:
    p + p
except TypeError as err:
    print("TypeError:", err)

# %%
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec:
    """The vector as a data class: __init__, __repr__, __eq__ and __hash__ are generated."""

    x: float
    y: float

    def __add__(self, other: "Vec") -> "Vec":
        return Vec(self.x + other.x, self.y + other.y)

    def __mul__(self, k: float) -> "Vec":
        if not isinstance(k, (int, float)):
            return NotImplemented  # Python then tries k.__rmul__, and raises TypeError if that fails
        return Vec(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __abs__(self) -> float:
        return math.hypot(self.x, self.y)

    def __matmul__(self, other: "Vec") -> float:  # u @ v: the dot product
        return self.x * other.x + self.y * other.y

    def __sub__(self, other: "Vec") -> "Vec":
        raise NotImplementedError  # TODO

    def __neg__(self) -> "Vec":
        raise NotImplementedError  # TODO

    def normalized(self) -> "Vec":
        """The unit vector with the same direction."""
        raise NotImplementedError  # TODO


a = Vec(3.0, 4.0)
b = Vec(1.0, 2.0)
print(a, a + b, 2 * a, abs(a), a @ b)
print(a == Vec(3.0, 4.0))
try:
    a * b
except TypeError as err:
    print("TypeError:", err)
try:
    a.x = 10.0
except AttributeError as err:  # FrozenInstanceError is a subclass of AttributeError
    print("frozen:", err)
print({a, Vec(3.0, 4.0), b})  # hashable, so usable in a set or as a dict key


# %%
@dataclass(frozen=True, order=True)
class OrderedVec:
    x: float
    y: float


vs = [OrderedVec(3.0, 4.0), OrderedVec(1.0, 2.0), OrderedVec(1.0, 10.0)]
print(sorted(vs))  # order=True compares (x, y) as tuples
print(sorted(vs, key=lambda v: math.hypot(v.x, v.y)))  # order by length
print(OrderedVec(1.0, 10.0) < OrderedVec(3.0, 4.0))

# %%
from collections import namedtuple

NTVec = namedtuple("NTVec", ["x", "y"])
n = NTVec(3.0, 4.0)
print(n, n.x, n[0], tuple(n))


# %%
def cosine_similarity(u: Vec, v: Vec) -> float:
    """The cosine of the angle between u and v."""
    raise NotImplementedError  # TODO


# print(cosine_similarity(Vec(1.0, 0.0), Vec(1.0, 1.0)))  # 0.7071...
