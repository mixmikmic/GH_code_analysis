import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
get_ipython().magic('matplotlib inline')

RTOL = 0.
ATOL = 1e-12
np.random.seed(17)

def is_square(a):
    return a.shape[0] == a.shape[1]

def has_solutions(a, b):
    return np.linalg.matrix_rank(a) == np.linalg.matrix_rank(np.append(a, b[np.newaxis].T, axis=1))

def is_dominant(a):
    return np.all(np.abs(a).sum(axis=1) <= 2 * np.abs(a).diagonal()) and         np.any(np.abs(a).sum(axis=1) < 2 * np.abs(a).diagonal())

def make_dominant(a):
    for i in range(a.shape[0]):
        a[i][i] = max(abs(a[i][i]), np.abs(a[i]).sum() - abs(a[i][i]) + 1)
    return a

def generate_random(n):
    return make_dominant(np.random.rand(n, n) * n), np.random.rand(n) * n

def generate_hilbert(n):
    return make_dominant(sp.linalg.hilbert(n)), np.arange(1, n + 1, dtype=np.float)

def linalg(a, b, debug=False):
    return np.linalg.solve(a, b),

def gauss(a, b, debug=False):
    assert is_square(a) and has_solutions(a, b)
    
    a = np.append(a.copy(), b[np.newaxis].T, axis=1)
    i = 0
    k = 0
    while i < a.shape[0]:
        r = np.argmax(a[i:, i]) + i
        a[[i, r]] = a[[r, i]]
        
        if a[i][i] == 0:
            break
        
        for j in range(a.shape[0]):
            if j == i:
                continue
            a[j] -= (a[j][i] / a[i][i]) * a[i]
        a[i] = a[i] / a[i][i]
        i += 1
        
    assert np.count_nonzero(a[i:]) == 0
    return a[:, -1],

def seidel(a, b, x0 = None, limit=20000, debug=False):
    assert is_square(a) and is_dominant(a) and has_solutions(a, b)
    
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float)
    x = x0.copy()
    while limit > 0:
        tx = x.copy()
        for i in range(a.shape[0]):
            x[i] = (b[i] - a[i, :].dot(x)) / a[i][i] + x[i]
        if debug:
            print(x)
        if np.allclose(x, tx, atol=ATOL, rtol=RTOL):
            return x, limit
        limit -= 1
    return x, limit

def jacobi(a, b, x0 = None, limit=20000, debug=False):
    assert is_square(a) and is_dominant(a) and has_solutions(a, b)
    
    if x0 is None:
        x0 = np.zeros_like(b, dtype=np.float)
    x = x0.copy()
    while limit > 0:
        tx = x.copy()
        for i in range(a.shape[0]):
            x[i] = (b[i] - a[i, :].dot(tx)) / a[i][i] + tx[i]
        if debug:
            print(x)
        if np.allclose(x, tx, atol=ATOL, rtol=RTOL):
            return x, limit
        limit -= 1
    return x, limit

def norm(a, b, res):
    return np.linalg.norm(a.dot(res) - b)

def run(method, a, b, verbose=False, **kwargs):
    if not verbose:
        print("-" * 100)
        print(method.__name__.upper())
    res = method(a, b, **kwargs)
    score = norm(a, b, res[0])
    if not verbose:
        print("res =", res)
        print("score =", score)
    return score

a4 = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])
print(a4)
b = np.array([6., 25., -11., 15.])
print("b =", b)

_ = run(linalg, a4, b)
_ = run(gauss, a4, b)
_ = run(seidel, a4, b)
_ = run(jacobi, a4, b)

a4 = np.array([[1., -1/8, 1/32, 1/64],
              [-1/2, 2., 1/16, 1/32],
              [-1., 1/4, 4., 1/16],
              [-1., 1/4, 1/8, 8.]])
print(a4)
b = np.array([1., 4., 16., 64.])
print("b =", b)

_ = run(linalg, a4, b)
_ = run(gauss, a4, b)
_ = run(seidel, a4, b)
_ = run(jacobi, a4, b)

a, b = generate_hilbert(1000)

print("LINALG =", run(linalg, a, b, verbose=True))
print("GAUSS =", run(gauss, a, b, verbose=True))
print("SEIDEL =", run(seidel, a, b, x0=np.zeros_like(b, dtype=np.float), verbose=True))
print("JACOBI =", run(jacobi, a, b, x0=np.zeros_like(b, dtype=np.float), verbose=True))

def plot_hilbert_score_by_matrix_size(method, sizes):
    scores = np.zeros_like(sizes, dtype=np.float)
    for i in range(len(sizes)):
        a, b = generate_hilbert(sizes[i])
        scores[i] = run(method, a, b, verbose=True)
    plt.plot(sizes, scores, label=method.__name__)

sizes = np.linspace(1, 600, num=50, dtype=np.int)

plt.figure(figsize=(15, 10))
plot_hilbert_score_by_matrix_size(linalg, sizes)
plot_hilbert_score_by_matrix_size(gauss, sizes)
plot_hilbert_score_by_matrix_size(seidel, sizes)
plot_hilbert_score_by_matrix_size(jacobi, sizes)

plt.title("Scores of different methods for Hilbert matrices")    .set_fontsize("xx-large")
plt.xlabel("n").set_fontsize("xx-large")
plt.ylabel("score").set_fontsize("xx-large")
legend = plt.legend(loc="upper right")
for label in legend.get_texts():
    label.set_fontsize("xx-large")
plt.show()

a, b = generate_random(20)

_ = run(linalg, a, b)
_ = run(gauss, a, b)
_ = run(seidel, a, b)
_ = run(jacobi, a, b)

a, b = generate_random(200)

get_ipython().magic('timeit run(linalg, a, b, verbose=True)')
get_ipython().magic('timeit run(gauss, a, b, verbose=True)')
get_ipython().magic('timeit run(seidel, a, b, verbose=True)')
get_ipython().magic('timeit run(jacobi, a, b, verbose=True)')

def plot_convergence(method, a, b, limits):
    scores = np.zeros_like(limits, dtype=np.float)
    for i in range(len(limits)):
        scores[i] = run(method, a, b, x0 = np.zeros_like(b, dtype=np.float), limit=limits[i], verbose=True)
    plt.plot(limits, scores, label=method.__name__)

a, b = generate_random(15)
limits = np.arange(0, 350)

plt.figure(figsize=(15, 10))
plot_convergence(seidel, a, b, limits)
plot_convergence(jacobi, a, b, limits)
plt.title("Convergence of Seidel/Jacobi methods for random matrix").set_fontsize("xx-large")
plt.xlabel("n_iters").set_fontsize("xx-large")
plt.ylabel("score").set_fontsize("xx-large")
plt.xscale("log")
legend = plt.legend(loc="upper right")
for label in legend.get_texts():
    label.set_fontsize("xx-large")
plt.show()

