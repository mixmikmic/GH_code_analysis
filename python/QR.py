import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from numba import jit

n = 500
m = 1000
A = np.zeros((m, n), dtype=np.complex128)
A.real = np.random.randn(m, n)
A.imag = np.random.randn(m, n)
# A = np.random.rand(m, n)

Q, R = np.linalg.qr(A)
np.linalg.norm(A - Q.dot(R))

print(Q.shape)
print(R.shape)

def QR_MGS(A):
    m, n = A.shape
    if m <= n:
        Q = np.zeros((m, m), dtype=A.dtype)
        R = np.zeros_like(A)
    else:
        Q = np.zeros_like(A)
        R = np.zeros((n, n), dtype=A.dtype)
    V = A.copy()
    for i in range(min(m, n)):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = Q[:, i].conj().dot(V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R

Q_mgs, R_mgs = QR_MGS(A)
print(Q_mgs.shape, R_mgs.shape)
print(np.linalg.norm(A - Q_mgs.dot(R_mgs)))

get_ipython().run_line_magic('timeit', 'QR_MGS(A)')

plt.imshow(R_mgs.real != 0)
plt.colorbar()

np.linalg.norm(Q_mgs.conj().T.dot(Q_mgs).real - np.eye(Q_mgs.shape[1]))

def QR_House(A):
    m, n = A.shape
    Q = np.eye(m, dtype=A.dtype)
    R = A.copy()
    for i in range(min(n, m)):
        y = R[i:, i].copy()
        s = y[0] / np.linalg.norm(y[0])
        norm = np.linalg.norm(y)
        y[0] += s * norm
        w = y / np.linalg.norm(y)
        w = np.reshape(w, (w.shape[0], 1))
        wTR = w.conj().T.dot(R[i:, i:])
        R[i:, i:] = R[i:, i:] - 2 * w.dot(wTR)
        Qw = Q[:, i:].dot(w)
        Q[:, i:] = Q[:, i:] - 2 * Qw.dot(w.conj().T) 
    return Q, R

Q_h, R_h = QR_House(A)
print(Q_h.shape, R_h.shape)
print(np.linalg.norm(Q_h.dot(R_h) - A))

get_ipython().run_line_magic('timeit', 'QR_House(A)')

plt.imshow(abs(R_h.real) > 1e-10)
plt.colorbar()

np.linalg.norm(Q_h.conj().T.dot(Q_h).real - np.eye(Q_h.shape[1]))







