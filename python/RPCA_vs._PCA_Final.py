import pickle as pkl
import numpy as np
from auto_bench.results import (bench_report,
                                compare_results,
                                crossval_plot)
from auto_bench.main import metric_test, best_param_combo

exp_dir = 'rpca_denoise/'
with open('results/' + exp_dir + 'rpca_denoise_param_map.pkl', 'r') as f:
    pmap_rpca = pkl.load(f)
with open('results/' + exp_dir + 'rpca_denoise.pkl', 'r') as f:
    R_rpca = pkl.load(f)
with open('results/' + exp_dir + 'opti_crossval.pkl', 'r') as f:
    results_rpca = pkl.load(f)

exp_dir = 'pca_denoise_full/'
with open('results/' + exp_dir + 'pca_denoise_param_map.pkl', 'r') as f:
    pmap_pca = pkl.load(f)
with open('results/' + exp_dir + 'pca_denoise.pkl', 'r') as f:
    R_pca = pkl.load(f)
with open('results/' + exp_dir + 'opti_crossval.pkl', 'r') as f:
    results_pca = pkl.load(f)
    
exp_dir = 'no_denoise_full/'
with open('results/' + exp_dir + 'default_no_denoise_param_map.pkl', 'r') as f:
    pmap_no = pkl.load(f)
with open('results/' + exp_dir + 'default_no_denoise.pkl', 'r') as f:
    R_no = pkl.load(f)
with open('results/' + exp_dir + 'opti_crossval.pkl', 'r') as f:
    results_no = pkl.load(f)

compare_results(results_no[0],
                'No Denoise')

compare_results(results_pca[0],
                'PCA Denoise')

compare_results(results_rpca[0][:-2],
                'RPCA Denoise')

pca_res, pca_labs = zip(*results_pca[0])
no_res, no_labs = zip(*results_no[0])
diff_res = np.array(pca_res) - np.array(no_res)
compare_results(zip(diff_res, no_labs),
                'Improvement of Disc. after PCA Denoise')

rpca_res, pca_labs = zip(*results_rpca[0][:-2])
no_res, no_labs = zip(*results_no[0][:-2])
diff_res = np.array(rpca_res) - np.array(no_res)
compare_results(zip(diff_res, no_labs),
                'Improvement of Disc. after RPCA Denoise')

rpca_res, pca_labs = zip(*results_rpca[0][:-2])
pca_res, pca_labs = zip(*results_pca[0][:-2])
diff_res = np.array(rpca_res) - np.array(pca_res)
compare_results(zip(diff_res, pca_labs),
                'How much better is RPCA')



