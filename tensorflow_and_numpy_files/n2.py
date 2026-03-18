import numpy as np

x = np.array([23.0, 24.0, 25.0, 26.0, 27.0], dtype=np.float64)
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"number of bytes in v: {v.nbytes}")
bits = v.nbytes * 8
print(f"number of total bits in v: {bits}")

r = np.array(([34.0,21.0,22.0,23.0,24.0], [35.0,25.0,26.0,27.0,28.0]))
print(r[0,-2])
print(r[1,-2])

print(r[0,:])
print(r[1,:])
print(r[0, 1:-2:2])

r[1,3] = 50
print(r)

a = np.array([[1,2,3,4,5],
              [6,7,8,9,10],
              [11,12,13,14,16]])
print(f"number of dimensions: {a.ndim}")
print(a[:, 2])

print(np.zeros((1,2,4,6)))

print(np.ones((1,2,4,6)))

print(np.full(2,2))

print(np.full(a.shape, 4))

print(np.random.rand(4,2))

arr = np.array([[1,2,3]])
y1 = np.repeat(arr, 3, axis=0)
print(y1)
print(y1.ndim)