get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

neural_data_df = pd.read_csv('data/darmanis_et_al_expressed_tf_fetal_neurons_scVLM_corrected.csv', sep='\t', 
                             index_col=0)
# The data has 5 columns of metadata and then the expression values. Genes are columns, rows are samples
neural_data_df

import scimitar.models
import scimitar.plotting
from collections import defaultdict



sns.set_style('white')
sns.set_context('talk', font_scale=2)

# For illustration purposes, we're using number of bootstrap iterations (n_boot) to 20. A reasonable number would be >100
def get_metastable_graph(data_array, covariance_type='diag'):
    results = scimitar.models.get_gmm_bootstrapped_metastable_graph(data_array, n_boot=20,
                                                                    covariance_type=covariance_type)
    metastable_graph, bootstrap_replicates , edge_fractions = results
    metastable_graph.edge_weights = edge_fractions
    
    plt.figure(1, figsize=(10,10))
    plt.clf()
    membership_colors, embedding = scimitar.plotting.plot_metastable_graph(data_array, metastable_graph, 
                                                                           edge_weights=edge_fractions)
    membership_list_dict = defaultdict(list)
    membership_array_dict = {}
    for i, color in enumerate(membership_colors):
        membership_list_dict[color].append(i)
    for color, lst in membership_list_dict.iteritems():
        membership_array_dict[color] = np.array(lst)
    return metastable_graph, membership_array_dict, membership_colors, embedding


# Here and throughout, we will extract the expression data to the neural_data_M array

neural_data_M = neural_data_df.iloc[:, 5:].values
ms_graph, membership_dict, membership_colors, embedding = get_metastable_graph(neural_data_M)

transition_model, analyzed_indices = ms_graph.fit_transition_model(neural_data_M, states=['blue', 'green', 'red'])
# analyzed_indices are indices of the states specified, in this case we don't really need them since we specified all
# states....the states argument can also be kept to its default, None, which selects all states

scimitar.plotting.plot_transition_model(neural_data_M,
                                    transition_model, 
                                    colors=membership_colors, 
                                    embedding=embedding,
                                    scatter_plot_args={'s':200, 'alpha':0.6},
                                    plot_errors=False)

import scimitar.morphing_mixture as mm
transition_model = mm.morphing_gaussian_from_embedding(neural_data_M, fit_type='spline', degree=3, step_size=0.07,
                                                      cov_estimator='corpcor', cov_reg=0.05)

refined_transition_model, refined_pseudotimes = transition_model.refine(neural_data_M, max_iter=3, step_size=0.07,
                                                                       cov_estimator='corpcor', cov_reg=0.05)


scimitar.plotting.plot_transition_model(neural_data_M,
                                    refined_transition_model, 
                                    colors=membership_colors, 
                                    embedding=embedding, 
                                    scatter_plot_args={'s':200, 'alpha':0.6},
                                    plot_errors=True, n_levels=10)

import dill
dill.dump(refined_transition_model, open('quake_neuron_transition_model.pickle', 'w'))
dill.dump(refined_pseudotimes, open('quake_neuron_pseudotimes.pickle', 'w'))

timepoints = np.arange(0, 1., 0.01)
means = refined_transition_model.mean(timepoints)
covs = refined_transition_model.covariance(timepoints)

import scimitar.differential_analysis
n_genes = neural_data_M.shape[1]
variances = np.array([covs[:, i, i] for i in xrange(n_genes)]).T
pvals = scimitar.differential_analysis.progression_association_lr_test(neural_data_M, means, variances)
#pvals = scimitar.differential_analysis.progression_association_f_test(neural_data_M, refined_transition_model,
#                                                                      refined_pseudotimes)

corrected_pvals, siggenes = scimitar.differential_analysis.p_adjust(pvals, correction_method='BH', threshold=0.05)

gene_names = neural_data_df.columns[5:]
cluster_members = scimitar.plotting.plot_transition_clustermap(means, gene_names, timepoints, n_clusters=5)

import scimitar.coexpression
corr_matrices = scimitar.coexpression.get_correlation_matrices(covs) 

from scipy.spatial.distance import pdist, squareform

n_matrices = corr_matrices.shape[0]
similarity_matrix = np.zeros([n_matrices, n_matrices])
for i in xrange(n_matrices):
    for j in xrange(i + 1, n_matrices):
        mat1 = abs(corr_matrices[i, :, :])
        mat2 = abs(corr_matrices[j, :, :])
        similarity_matrix[i, j] = np.linalg.norm(mat1 - mat2)
        similarity_matrix[j, i] = similarity_matrix[i, j]
        

# Make it a proper similarity matrix, normalize to one
similarity_matrix /= similarity_matrix.max()
similarity_matrix = 1 - similarity_matrix

# Visulize it
sns.set_context('talk', font_scale=2)
sns.heatmap(similarity_matrix, cmap=plt.get_cmap('Blues'))
plt.xticks([])
plt.yticks([])
plt.xlabel('Pseudotime')
plt.ylabel('Pseudotime')

from sklearn.cluster import SpectralClustering
spectral = SpectralClustering(n_clusters=3, affinity='precomputed')
labels = spectral.fit_predict(similarity_matrix)

coreg_states = {}
for ci in np.unique(labels):
    coreg_states[ci] = corr_matrices[labels == ci, :, :].mean(axis=0)


for ci, coreg_state in coreg_states.iteritems():
    cm = sns.clustermap(coreg_state)
    plt.title(ci)
    cm.ax_heatmap.set_xticks([])
    cm.ax_heatmap.set_yticks([])

