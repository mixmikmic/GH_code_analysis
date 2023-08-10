import numpy as np, GPy, pandas as pd
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns

from topslam.simulation import qpcr_simulation

seed_differentiation = 5001
seed_gene_expression = 0

Xsim, simulate_new, t, c, labels, seed = qpcr_simulation(seed=seed_differentiation)

np.random.seed(seed_gene_expression)
Y = simulate_new()

from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding, Isomap
from sklearn.decomposition import FastICA, PCA

methods = {'t-SNE':TSNE(n_components=2),
           'PCA':PCA(n_components=2), 
           'Spectral': SpectralEmbedding(n_components=2, n_neighbors=10), 
           'Isomap': Isomap(n_components=2, n_neighbors=10), 
           'ICA': FastICA(n_components=2)
           }

from topslam.optimization import run_methods

X_init, dims = run_methods(Y, methods)

from topslam.optimization import create_model, optimize_model
m = create_model(Y, X_init, linear_dims=0)

m.optimize(messages=1, max_iters=5e3)

m.kern.plot_ARD()
print("Most significant input dimensions: {}".format(m.kern.get_most_significant_input_dimensions()[:2]))

m.plot_scatter(labels=labels)

from topslam import ManifoldCorrectionTree
m_topslam = ManifoldCorrectionTree(m)

ax = m_topslam.plot_waddington_landscape()
m_topslam.plot_time_graph(labels=labels, start=0, ax=ax)

from topslam.plotting import plot_comparison
with sns.axes_style('white'):
    ax = plot_comparison(m_topslam, X_init, dims, labels, np.unique(labels), 0)

from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
from topslam.simulation.graph_extraction import extract_manifold_distances_mst

pt_topslam = m_topslam.get_pseudo_time(start=0)
D_ica, mst_ica = extract_manifold_distances_mst(squareform(pdist(X_init[:, dims['ICA']])))
pt_ica = D_ica[0] # only take the distances from the start (which is 0 in this case)
D_tsne, mst_tsne = extract_manifold_distances_mst(squareform(pdist(X_init[:, dims['t-SNE']])))
pt_tsne = D_tsne[0] # only take the distances from the start (which is 0 in this case)

result_df = pd.DataFrame(columns=['slope', 'intercept', 'r_value', 'p_value', 'std_err'])
result_df.loc['topslam'] = linregress(pt_topslam, t.flat)[:]
result_df.loc['ICA'] = linregress(pt_ica, t.flat)[:]
result_df.loc['t-SNE'] = linregress(pt_tsne, t.flat)

result_df

time_df = pd.DataFrame(t, columns=['simulated'])
time_df['topslam'] = pt_topslam
time_df['ICA'] = pt_ica
time_df['t-SNE'] = pt_tsne

sns.jointplot('simulated', 'topslam', data=time_df, kind='reg')

sns.jointplot('simulated', 'ICA', data=time_df, kind='reg')

sns.jointplot('simulated', 't-SNE', data=time_df, kind='reg')



