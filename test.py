import timeit
import numpy as np

a = np.random.rand(1000000).astype(np.float32)
b = 5.0

# Division
def with_div():
    return a / b

# Multiplication by reciprocal
def with_mult():
    return a * (1.0 / b)

print("Division:", timeit.timeit(with_div, number=100))
print("Multiplication:", timeit.timeit(with_mult, number=100))
