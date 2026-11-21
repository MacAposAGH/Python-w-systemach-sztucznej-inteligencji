import math
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

tangent_points_range = 10
tangent_points = [i * (2 * math.pi / (tangent_points_range - 1)) for i in range(tangent_points_range)]

points_range = 100
x = [i * (2 * math.pi / (points_range - 1)) for i in range(points_range)]

y = [math.sin(i) for i in x]

plt.plot(x, y, label='sin(x)')
for i, x0 in enumerate(tangent_points):
    slope = math.cos(x0)
    plt.plot(x, [math.sin(x0) + slope * (xs - x0) for xs in x], '--')

plt.grid()
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
