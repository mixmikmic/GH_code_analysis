get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import ipy_utils
import numpy as np
import pandas as pd

GO_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')

#GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('results_all_tissues')
GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('Results/full_results_all_tissues_loss_l1_neg_0')
GO_gene_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')
aucs_prob = ipy_utils.make_roc_curves(GO_terms, GO_gene_counts)
print np.mean(aucs_prob), np.mean(roc_aucs_strat)

GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('Results/pca_results_all_tissues_loss_l1_neg_0')
GO_gene_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')
aucs_prob = ipy_utils.make_roc_curves(GO_terms, GO_gene_counts)
print np.mean(aucs_prob), np.mean(roc_aucs_strat)

GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('Results/median_results_all_tissues_loss_l1_neg_0')
GO_gene_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')
aucs_prob = ipy_utils.make_roc_curves(GO_terms, GO_gene_counts)
print np.mean(aucs_prob), np.mean(roc_aucs_strat)

plt.hist(aucs_prob)
plt.xlim([0.0, 1.0])
plt.xlabel('ROC AUC')
plt.ylabel('# of GO Terms')
plt.show()



