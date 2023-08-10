get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from IPython.display import display, Math, Latex, Markdown

def plot_cov_ellipse(data, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        W, V = np.linalg.eig(cov)
        order = W.argsort()[::-1]
        return W[order], V[:,order]
    
    cov = np.cov(data, rowvar=False)

    if ax is None:
        ax = plt.gca()

    W, V = eigsorted(cov)
    display(Markdown('# Eigenvalue decomposition #'))
    print("eigen vals:", W)
    print("eigen vectors:\n", V)
    theta = np.degrees(np.arctan2(*V[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(W)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    ax.set_title('From eigenvalues')
    return ellip

def plot_svd_ellipse(data, pos, nstd=2, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    centered_data = data - pos
    U, S, V_T = np.linalg.svd(centered_data)
    V = V_T.T
    display(Markdown('# SVD decomposition #'))
    print("singular vals:", S)
    print("singular vectors:\n", V)
    theta = np.degrees(np.arctan2(*V[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * (S / np.sqrt(data.shape[0]))
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    ax.set_title('From singular values')
    return ellip

def compare_eigen_svd(data):
    display(Markdown('# Eigenvalues vs singular values of a cov matrix #'))
    display(Markdown('The eigenvalues should be the same as singular values '
                     'when `eig` and `SVD` are applied to a covariance matrix $\Sigma$'))
    cov = np.cov(data, rowvar=False)
    W, V_eig = np.linalg.eig(cov)
    U, S, V_svd = np.linalg.svd(cov)
    display(Markdown('### Eigenvalues of $\Sigma$'))
    print(W)
    display(Markdown('### Singular values of $\Sigma$'))
    print(S)

# Generate some random, correlated data
points = np.random.multivariate_normal(mean=(20,10), cov=[[18, 12],[12, 30]], size=1000)
pos = points.mean(axis=0)
plot_size = (6,6)
x, y = points.T
fig = plt.figure(0, figsize=plot_size)
ax = fig.add_subplot(111)
ax.plot(x, y, 'bo')
plot_cov_ellipse(points, pos, nstd=3, alpha=0.2, color='blue')
plt.show()

fig = plt.figure(1, figsize=plot_size)
ax = fig.add_subplot(111)
ax.plot(x, y, 'bo')
plot_svd_ellipse(points, pos, nstd=3, alpha=0.5, color='orange')
plt.show()

# compare eigen values and singular values of a cov matrix
compare_eigen_svd(points)

