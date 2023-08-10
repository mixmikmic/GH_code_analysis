import numpy as np

n = 100000
xs = np.random.random(n)
ys = np.random.random(n)

ds = np.hypot(xs, ys)

hits = (ds < 1-xs) & (ds < 1-ys)

hits.mean()

xs = np.linspace(0, 1, 1000)
ys = xs
X, Y = np.meshgrid(xs, ys)

def in_region(X, Y):
    """Compute the fraction of points in the region of interest.
    
    X: array of x coordinates
    Y: array of y coordinates
    
    returns: float
    """
    ds = np.hypot(X, Y)
    hits = (ds < 1-X) & (ds < 1-Y)
    return hits.mean()

in_region(X, Y)

(4 * np.sqrt(2) - 5) / 3

X.nbytes

xs = np.linspace(0, 1, 1000)
ys = xs
X = xs[:, None]
Y = ys[None, :]

X.nbytes

in_region(X, Y)

