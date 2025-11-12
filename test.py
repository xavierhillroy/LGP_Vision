import timeit
import numpy as np

high = 4
low = 0 
rng = np.random.default_rng()
b = rng.integers(low,high, dtype=np.int16,)


cut = 2 
list_1 = [1,2,3,4]

list_2 = [5,6,7,8]
max_cut = min(len(list_1),len(list_2))- 1
# print(max_cut)
cut = max_cut
temp = list_1 
list_1 = list_1[:cut] + list_2[cut:]
list_2 = list_2[:cut] + temp[cut:]
# c = list_1[:cut] + list_2[cut:]
# c2 = list_2[:cut] + list_1[cut:]

# print(c)
# print(c2)
# print(list_1)
# print(list_2)
p1 = [1,2,3,4,5]
p2 =[6,7,8,9]
c11 = 1
c12 = 2
c21 = 1
c22 = 3 

max_prog_len = 10
# c11 = rng.integers(0, len(p1))
# c12 = rng.integers(0, len(p1))
# c21 = rng.integers(0, len(p2))
# c22 = rng.integers(0, len(p2))
c1 = p1[:c11] + p2[c21:c22] + p1[c12:]
c2 = p2[:c21] + p1[c11:c12] + p2[c22:]
print(c1)
print(c2)
print(p2[c21:c22])
print(list_1[1:2])
# print(rng.integers(0,))