import os

HOME = os.path.expanduser('~')

DATA_FOLDER = os.path.abspath(
    os.path.join(HOME, 'projects', 'cshl-singlecell-2017', 'data'))
FIGURE_FOLDER = os.path.abspath(
    os.path.join(HOME, 'projects', 'cshl-singlecell-2017', 'figures'))


notebook_name = '50_Example_workflow_reanalyzing_macosko2015'
data_folder = os.path.join(DATA_FOLDER, notebook_name)
figure_folder = os.path.join(FIGURE_FOLDER, notebook_name)

input_folder = os.path.join(DATA_FOLDER, '91_filter_genes')

get_ipython().system(' mkdir -p $data_folder')
get_ipython().system(' mkdir -p $figure_folder')

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, MDS

import pandas as pd
import matplotlib.pyplot as plt
import phenograph

import macosko2015
get_ipython().magic('matplotlib inline')

counts, cells, genes = macosko2015.load_differential_clusters()
print('counts.shape', counts.shape)
print('cells.shape', cells.shape)
print('genes.shape', genes.shape)

counts.head()

genes.head()

cells.head()

pcaer = PCA(n_components=15)
# pcad = pcaer.fit_tra

pcad = pcaer.fit_transform(counts)
pcad

pcad.shape

pcad_df = pd.DataFrame(pcad, index=counts.index)
print(pcad_df.shape)
pcad_df.head()

get_ipython().run_cell_magic('time', '', '\nsmusher = TSNE()\ntsned = smusher.fit_transform(pcad_df)\nprint(tsned.shape)\ntsned')



