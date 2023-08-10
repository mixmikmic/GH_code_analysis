get_ipython().run_line_magic('run', '../__init__.py')

db_pca_results = pd.read_pickle('../Datasets/db_pca_results.p')
db_skb_results = pd.read_pickle('../Datasets/db_skb_results.p') 
db_sfm_results = pd.read_pickle('../Datasets/db_sfm_results.p')                               
uci_pca_results = pd.read_pickle('../Datasets/uci_pca_scores.p')
uci_skb_results = pd.read_pickle('../Datasets/uci_skb_scores.p') 
uci_sfm_results = pd.read_pickle('../Datasets/uci_sfm_scores.p')                               

db_pca_results['Method']='PCA'
db_skb_results['Method']='SKB'
db_sfm_results['Method']='SFM'

db_results = pd.concat([db_pca_results, db_skb_results, db_sfm_results], ignore_index=True)

db_top_5 = db_results.sort_values(['test_score'], ascending=False).head(5)

db_top_5

db_top_5.to_pickle('../Datasets/db_top5.p')

db_top_5['best_params'][9]

uci_pca_results['Method']='PCA'
uci_skb_results['Method']='SKB'
uci_sfm_results['Method']='SFM'

uci_results = pd.concat([uci_pca_results, uci_skb_results, uci_sfm_results], ignore_index=True)

uci_top_5 = uci_results.sort_values(['test_score'], ascending=False).head(5)

uci_top_5

uci_top_5.to_pickle('../Datasets/uci_top5.p')

uci_top_5['best_params'][0]

