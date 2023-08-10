get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import ipy_utils

GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('pca_results_all_tissues_0')
# 'pca_results_1_tissue_loss_l2_neg_0_server_0')
GO_gene_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')
aucs_prob = ipy_utils.make_roc_curves(GO_terms, GO_gene_counts)
print np.mean(aucs_prob), np.mean(roc_aucs_strat), len(aucs_prob)

GO_terms, roc_aucs_strat = ipy_utils.get_prediction_results('pca_results_all_tissues_loss_l2_neg_0')
GO_gene_counts = ipy_utils.get_GO_gene_counts('../data/GO_terms_final_gene_counts.txt')
aucs_prob = ipy_utils.make_roc_curves(GO_terms, GO_gene_counts)
print np.mean(aucs_prob), np.mean(roc_aucs_strat), len(aucs_prob)



