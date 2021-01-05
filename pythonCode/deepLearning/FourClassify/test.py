import numpy as np

M = np.ones((5,5))
M = np.array([M])
print(np.shape(M))
N = np.repeat(M,3,axis=0)
N = N.transpose(1,2,0)
print(N)
print(np.shape(N))