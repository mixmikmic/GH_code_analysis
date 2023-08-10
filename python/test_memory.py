get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

get_ipython().run_cell_magic('writefile', 'embed.py', '"""Generate a diffusion map embedding\n"""\n\ndef compute_diffusion_map(L, alpha=0.5, n_components=None, diffusion_time=0,\n                          skip_checks=False, overwrite=False):\n    """Compute the diffusion maps of a symmetric similarity matrix\n\n        L : matrix N x N\n           L is symmetric and L(x, y) >= 0\n\n        alpha: float [0, 1]\n            Setting alpha=1 and the diffusion operator approximates the\n            Laplace-Beltrami operator. We then recover the Riemannian geometry\n            of the data set regardless of the distribution of the points. To\n            describe the long-term behavior of the point distribution of a\n            system of stochastic differential equations, we can use alpha=0.5\n            and the resulting Markov chain approximates the Fokker-Planck\n            diffusion. With alpha=0, it reduces to the classical graph Laplacian\n            normalization.\n\n        n_components: int\n            The number of diffusion map components to return. Due to the\n            spectrum decay of the eigenvalues, only a few terms are necessary to\n            achieve a given relative accuracy in the sum M^t.\n\n        diffusion_time: float >= 0\n            use the diffusion_time (t) step transition matrix M^t\n\n            t not only serves as a time parameter, but also has the dual role of\n            scale parameter. One of the main ideas of diffusion framework is\n            that running the chain forward in time (taking larger and larger\n            powers of M) reveals the geometric structure of X at larger and\n            larger scales (the diffusion process).\n\n            t = 0 empirically provides a reasonable balance from a clustering\n            perspective. Specifically, the notion of a cluster in the data set\n            is quantified as a region in which the probability of escaping this\n            region is low (within a certain time t).\n\n        skip_checks: bool\n            Avoid expensive pre-checks on input data. The caller has to make\n            sure that input data is valid or results will be undefined.\n\n        overwrite: bool\n            Optimize memory usage by re-using input matrix L as scratch space.\n\n        References\n        ----------\n\n        [1] https://en.wikipedia.org/wiki/Diffusion_map\n        [2] Coifman, R.R.; S. Lafon. (2006). "Diffusion maps". Applied and\n        Computational Harmonic Analysis 21: 5-30. doi:10.1016/j.acha.2006.04.006\n    """\n\n    import numpy as np\n    import scipy.sparse as sps\n\n    use_sparse = False\n    if sps.issparse(L):\n        use_sparse = True\n\n    if not skip_checks:\n        from sklearn.manifold.spectral_embedding_ import _graph_is_connected\n        if not _graph_is_connected(L):\n            raise ValueError(\'Graph is disconnected\')\n\n    ndim = L.shape[0]\n    if overwrite:\n        L_alpha = L\n    else:\n        L_alpha = L.copy()\n\n    if alpha > 0:\n        # Step 2\n        d = np.array(L_alpha.sum(axis=1)).flatten()\n        d_alpha = np.power(d, -alpha)\n        if use_sparse:\n            L_alpha.data *= d_alpha[L_alpha.indices]\n            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())\n            L_alpha.data *= d_alpha[L_alpha.indices]\n            L_alpha = sps.csr_matrix(L_alpha.transpose().toarray())\n        else:\n            L_alpha = d_alpha[:, np.newaxis] * L_alpha \n            L_alpha = L_alpha * d_alpha[np.newaxis, :]\n\n    # Step 3\n    d_alpha = np.power(np.array(L_alpha.sum(axis=1)).flatten(), -1)\n    if use_sparse:\n        L_alpha.data *= d_alpha[L_alpha.indices]\n    else:\n        L_alpha = d_alpha[:, np.newaxis] * L_alpha\n\n    M = L_alpha\n\n    from scipy.sparse.linalg import eigsh, eigs\n\n    # Step 4\n    func = eigs\n    if n_components is not None:\n        lambdas, vectors = func(M, k=n_components + 1)\n    else:\n        lambdas, vectors = func(M, k=max(2, int(np.sqrt(ndim))))\n    del M\n\n    if func == eigsh:\n        lambdas = lambdas[::-1]\n        vectors = vectors[:, ::-1]\n    else:\n        lambdas = np.real(lambdas)\n        vectors = np.real(vectors)\n        lambda_idx = np.argsort(lambdas)[::-1]\n        lambdas = lambdas[lambda_idx]\n        vectors = vectors[:, lambda_idx]\n\n    # Step 5\n    psi = vectors/vectors[:, [0]]\n    if diffusion_time == 0:\n        lambdas = lambdas[1:] / (1 - lambdas[1:])\n    else:\n        lambdas = lambdas[1:] ** float(diffusion_time)\n    lambda_ratio = lambdas/lambdas[0]\n    threshold = max(0.05, lambda_ratio[-1])\n\n    n_components_auto = np.amax(np.nonzero(lambda_ratio > threshold)[0])\n    n_components_auto = min(n_components_auto, ndim)\n    if n_components is None:\n        n_components = n_components_auto\n    embedding = psi[:, 1:(n_components + 1)] * lambdas[:n_components][None, :]\n\n    result = dict(lambdas=lambdas, vectors=vectors,\n                  n_components=n_components, diffusion_time=diffusion_time,\n                  n_components_auto=n_components_auto)\n    return embedding, result')

from embed import compute_diffusion_map

def compute_affinity(X, method='markov', eps=None):
    import numpy as np
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, metric='euclidean')
    if eps is None:
        k = int(max(2, np.round(D.shape[0] * 0.01)))
        eps = 2 * np.median(np.sort(D, axis=0)[k+1, :])**2
    if method == 'markov':
        affinity_matrix = np.exp(-(D * D) / eps)
    elif method == 'cauchy':
        affinity_matrix = 1./(D * D + eps)
    return affinity_matrix

import numpy as np
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer

n=5000
t=np.power(np.sort(np.random.rand(n)), .7)*10
al=.15;bet=.5;
x1=bet * np.exp(al * t) * np.cos(t) + 0.1 * np.random.randn(n)
y1=bet * np.exp(al * t) * np.sin(t) + 0.1 * np.random.randn(n)
X = np.hstack((x1[:, None], y1[:, None]))

plt.scatter(x1, y1, c=t, cmap=plt.cm.Spectral, linewidths=0)
ph = plt.plot(x1[0], y1[0], 'ko')

L = compute_affinity(X.copy(), method='markov')

import scipy.sparse as sps

get_ipython().run_line_magic('load_ext', 'memory_profiler')

get_ipython().run_line_magic('mprun', '-f compute_diffusion_map compute_diffusion_map(L)')

get_ipython().run_line_magic('memit', 'compute_diffusion_map(L, skip_checks=True, overwrite=True)')



