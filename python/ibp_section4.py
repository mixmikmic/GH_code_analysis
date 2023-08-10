import numpy as np
import math
import time

N = 3000
B = np.random.randn(N, N)
A = B.T.dot(B)
print(A.shape)
print(A[:2,:2])
start = time.time()
L = np.linalg.cholesky(A)
print('time for full rank Cholesky %s seconds' % (time.time() - start))
print(L[:3,:3])

# check
LLT = L.dot(L.T)
assert np.abs(A - LLT).max() < 1e-6

x = np.random.randn(N)
print(x.shape)
print(x[:3])

def update_cholesky(L, x):
    N = x.shape[0]
    for k in range(N):
        r = math.sqrt(L[k, k] * L[k, k] + x[k] * x[k])
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        L[k + 1:, k] = (L[k + 1:, k] + s * x[k + 1:]) / c
        x[k + 1:] = c * x[k + 1:] - s * L[k + 1:, k]

update_cholesky(L, x)
LLT2 = L.dot(L.T)
diff = np.abs(A + x.dot(x.T) - LLT2).max()
print(diff)
assert diff < 1e-8

import numpy as np
import math
import time

N = 3000
B = np.random.randn(N, N)
A = B.T.dot(B)
print(A.shape)
print(A[:2,:2])
start = time.time()
Ainv = np.linalg.inv(A)
print('time for full rank inverse %s seconds' % (time.time() - start))
print(Ainv[:3,:3])

# check
A_Ainv = A.dot(Ainv)
diff = np.abs(A_Ainv - np.identity(N)).max()
print(diff)
assert diff < 1e-8

x = np.random.randn(N, 1)
print(x.shape)
print(x[:3])

def sherman_morrison(Ainv, x):
    numerator = Ainv.dot(x).dot(x.T).dot(Ainv)
    denominator = x.T.dot(Ainv).dot(x) - 1
    return Ainv - numerator / denominator

start = time.time()
Ainv2 = sherman_morrison(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2 = A + x.dot(x.T)
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)
# assert diff < 1e-8

# try full rank inverse
start = time.time()
Ainv2 = np.linalg.inv(A2)
print('time for full rank inverse %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after full rank inverse %s' % diff)

def sherman_morrison_v2(Ainv, x):
    n1 = Ainv.dot(x)
    n2 = x.T.dot(Ainv)
    numerator = n1.dot(n2)
    denominator = x.T.dot(Ainv).dot(x) - 1
    return Ainv - numerator / denominator

start = time.time()
Ainv2 = sherman_morrison_v2(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2 = A + x.dot(x.T)
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)
# assert diff < 1e-8

# try full rank inverse
start = time.time()
Ainv2 = np.linalg.inv(A2)
print('time for full rank inverse %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after full rank inverse %s' % diff)

def sherman_morrison_v3(Ainv, x):
    n1 = Ainv.dot(x)
    numerator = n1.dot(n1.T)
    denominator = x.T.dot(Ainv).dot(x) - 1
    return Ainv - numerator / denominator

start = time.time()
Ainv2 = sherman_morrison_v3(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2 = A + x.dot(x.T)
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)
# assert diff < 1e-8

# try full rank inverse
start = time.time()
Ainv2 = np.linalg.inv(A2)
print('time for full rank inverse %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after full rank inverse %s' % diff)

def sherman_morrison_with_timing(Ainv, x):
    last = time.time()
    n1 = Ainv.dot(x)
    print('time for n1 %s' % (time.time() - last))
    last = time.time()
    numerator = n1.dot(n1.T)
    print('time for numerator %s' % (time.time() - last))
    last = time.time()
    denominator = x.T.dot(Ainv).dot(x) - 1
    print('time for denominator %s' % (time.time() - last))
    last = time.time()
    res = Ainv - numerator / denominator
    print('time for final per-element %s' % (time.time() - last))
    return res

start = time.time()
Ainv2 = sherman_morrison_with_timing(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)

def sherman_morrison_with_timing_v2(Ainv, x):
    last = time.time()
    n1 = Ainv.dot(x)
    print('time for n1 %s' % (time.time() - last))
    last = time.time()
    numerator = n1.dot(n1.T)
    print('time for numerator %s' % (time.time() - last))
    last = time.time()
    denominator = x.T.dot(Ainv).dot(x) - 1
    print('time for denominator %s' % (time.time() - last))
    last = time.time()
    res = Ainv - numerator * (1 / denominator)
    print('time for final per-element %s' % (time.time() - last))
    return res

start = time.time()
Ainv2 = sherman_morrison_with_timing_v2(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)

def sherman_morrison_with_timing_v2(Ainv, x):
    last = time.time()
    n1 = Ainv.dot(x)
    print('time for n1 %s' % (time.time() - last))
    last = time.time()
    numerator = n1.dot(n1.T)
    print('time for numerator %s' % (time.time() - last))
    last = time.time()
    denominator = x.T.dot(Ainv).dot(x) - 1
    print('time for denominator %s' % (time.time() - last))
    last = time.time()
    res = Ainv - numerator * (1 / denominator.item())
    print('time for final per-element %s' % (time.time() - last))
    return res

start = time.time()
Ainv2 = sherman_morrison_with_timing_v2(Ainv, x)
print('time for sherman morrison %s seconds' % (time.time() - start))
A2_Ainv2 = A2.dot(Ainv2)
diff = np.abs(A2_Ainv2 - np.identity(N)).max()
print('diff after rank 1 update: %s' % diff)

