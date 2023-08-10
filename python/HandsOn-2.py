get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import xarray as xr

from rankeval.dataset import Dataset
from rankeval.model import RTEnsemble
from rankeval.metrics import NDCG
from rankeval.analysis.effectiveness import tree_wise_performance
from rankeval.visualization.effectiveness import plot_tree_wise_performance
from rankeval.analysis.effectiveness import model_performance

# Global Options

# paths to executable files
QUICKRANK = "./quickrank/bin/quicklearn"
SCORER    = "./quickrank/bin/quickscore"

QUICKSCORER = "./QuickScorer/bin/quickscorer"

# paths to Istella-S dataset
train_dataset_file = "/data/letor-datasets/tiscali/sample/train.txt"
valid_dataset_file = "/data/letor-datasets/tiscali/sample/vali.txt"
test_dataset_file  = "/data/letor-datasets/tiscali/sample/test.txt"

# paths to model file
models_folder            = "models"
baseline_model_file      = os.path.join(models_folder, "istella-small.lamdamart.xml")
cleaver_model_file       = os.path.join(models_folder, "istella-small.lamdamart.cleaver.xml")

dart_model_file          = os.path.join(models_folder, "istella-small.dart.xml")
xdart_model_file         = os.path.join(models_folder, "istella-small.xdart.xml")

small_dart_model_file    = os.path.join(models_folder, "istella-small.dart.small.xml")
small_xdart_model_file   = os.path.join(models_folder, "istella-small.xdart.small.xml")

# setting floating point precision of Pandas
pd.set_option('precision', 4)

get_ipython().system('{QUICKRANK}     --train {train_dataset_file}     --valid {valid_dataset_file}     --model-out {baseline_model_file}     --num-trees 1500     --num-leaves 64     --shrinkage 0.05')

# load istella-S dataset
train_dataset = Dataset.load(train_dataset_file, name="train")
valid_dataset = Dataset.load(valid_dataset_file, name="valid")
test_dataset  = Dataset.load(test_dataset_file, name="test")

# load the model
baseline_model = RTEnsemble(baseline_model_file, name="LambdaMart", format="QuickRank")

# define metric
ndcg_10 = NDCG(cutoff=10, 
               no_relevant_results=0.0) # assign score 0 to queries without relevant docs

# measure NDCG every 20 trees
tree_wise_perf = tree_wise_performance( datasets =[train_dataset, valid_dataset, test_dataset], 
                                        models   =[baseline_model],
                                        metrics  =[ndcg_10],
                                        step=20)

fig_list = plot_tree_wise_performance(tree_wise_perf, compare = "datasets")

tree_wise_perf.loc[{'dataset':test_dataset}].to_dataframe()

baseline_effectiveness = tree_wise_perf.loc[{'dataset':test_dataset, 'model':baseline_model, 'metric':ndcg_10}].values[-1]

# We use Con-Op based C code as a baseline for the scoring time evaluation

def run_condop(model_file, dataset_file, rounds=1):
    # create the C code
    print (" 1. Creating the C code for " + model_file)
    condop_source = model_file + ".c"
    
    _ = get_ipython().getoutput('{QUICKRANK}       --generator condop       --model-file {model_file}       --code-file {condop_source}')
    
    # Compile an executable ranker. The resulting ranker is SCORER=./quickrank/bin/quickscore
    print (" 2. Compiling the model")

    # replace empty scorer
    get_ipython().system('cp {condop_source} ./quickrank/src/scoring/ranker.cc')
    # compile
    _ = get_ipython().getoutput('make -j -C ./quickrank/build_ quickscore ')
    
    # Run the compiled model
    print (" 3. Running the compiled model")
    scorer_out = get_ipython().getoutput('{SCORER}       -d {dataset_file}       -r {rounds}')
    
    print (scorer_out.n)
    
    # takes the scoring time in milli-seconds
    scoring_time = float(scorer_out.l[-1].split()[-2])* 10**6
    
    return scoring_time

baseline_efficiency = run_condop(baseline_model_file, test_dataset_file)

# Store current results
results = pd.DataFrame(columns=['Model', '# Trees', 'NDCG@10', 'Scoring Time µs.'])

results.loc[len(results)] = [baseline_model.name, baseline_model.n_trees, baseline_effectiveness, baseline_efficiency]
results

pruning_rate = 0.5
get_ipython().system('{QUICKRANK}    --model-in {baseline_model_file}     --train {train_dataset_file}     --valid {valid_dataset_file}     --opt-algo CLEAVER     --opt-method QUALITY_LOSS_ADV     --pruning-rate {pruning_rate}     --with-line-search     --num-samples 20     --window-size 10     --reduction-factor 0.95     --max-iterations 100     --max-failed-valid 20     --adaptive     --opt-algo-model {cleaver_model_file}')

cleaver_model = RTEnsemble(cleaver_model_file, name="Cleaver", format="QuickRank")

# measure NDCG
tree_wise_perf = tree_wise_performance(datasets=[test_dataset], 
                                        models=[baseline_model, cleaver_model],
                                        metrics=[ndcg_10],
                                        step=20)
fig_list = plot_tree_wise_performance(tree_wise_perf, compare = "models")

models_perf = model_performance(datasets=[test_dataset], 
                                 models=[cleaver_model, baseline_model], 
                                 metrics=[ndcg_10])
models_perf.to_dataframe()

from rankeval.analysis.statistical import statistical_significance
stat_sig = statistical_significance(datasets=[test_dataset],
                                    model_a=cleaver_model, model_b=baseline_model, 
                                    metrics=[ndcg_10],
                                    n_perm=100000 )
stat_sig.to_dataframe()

cleaver_effectiveness = float( models_perf.loc[{'model':cleaver_model, 'dataset':test_dataset, 'metric':ndcg_10}] )

cleaver_efficiency = run_condop(cleaver_model_file, test_dataset_file)

results.loc[len(results)] = [cleaver_model.name, cleaver_model.n_trees, cleaver_effectiveness, cleaver_efficiency]
results

# Train Dart with QuickRank
get_ipython().system('{QUICKRANK}   --algo DART   --train {train_dataset_file}   --valid {valid_dataset_file}   --model-out {dart_model_file}   --num-trees 1500   --num-leaves 64   --shrinkage 1.0   --sample-type UNIFORM   --normalize-type TREE   --adaptive-type FIXED   --rate-drop 0.015')

# load Dart Models
dart_model = RTEnsemble(dart_model_file, name="Dart", format="QuickRank")

# measure NDCG
tree_wise_perf = tree_wise_performance(datasets=[test_dataset], 
                                        models=[baseline_model, cleaver_model, dart_model],
                                        metrics=[ndcg_10],
                                        step=20)
fig_list = plot_tree_wise_performance(tree_wise_perf, compare = "models")

# we want at least the same effectiveness as the baseline model
tree_wise_perf.loc[{'model':dart_model, 'k':range(700,800,20)}].to_dataframe()

small_dart_trees = 760
small_dart = dart_model.copy(n_trees=small_dart_trees)
small_dart.save(small_dart_model_file)

small_dart_effectiveness = float( tree_wise_perf.loc[{'model':dart_model, 'k':small_dart_trees, 'metric':ndcg_10}].values )

small_dart_efficiency = run_condop(small_dart_model_file, test_dataset_file)

results.loc[len(results)] = [small_dart.name, small_dart.n_trees, small_dart_effectiveness, small_dart_efficiency]
results

# Train X-Dart with QuickRank
get_ipython().system('{QUICKRANK}   --algo DART   --train {train_dataset_file}   --valid {valid_dataset_file}   --model-out {xdart_model_file}   --num-trees 1500   --num-leaves 64   --shrinkage 1.0   --sample-type UNIFORM   --normalize-type TREE   --adaptive-type PLUSHALF_RESET_LB1_UB5   --rate-drop 1.0   --keep-drop   --drop-on-best')

xdart_model = RTEnsemble(xdart_model_file, name="X-Dart", format="QuickRank")

# measure NDCG
tree_wise_perf = tree_wise_performance(datasets=[test_dataset], 
                                        models=[baseline_model, cleaver_model, dart_model, xdart_model],
                                        metrics=[ndcg_10],
                                        step=20)
fig_list = plot_tree_wise_performance(tree_wise_perf, compare = "models")

# we want at least the same effectiveness as the baseline model
tree_wise_perf.loc[{'model':xdart_model, 'k':range(500,720,20)}].to_dataframe()

small_xdart_trees = 560
small_xdart = xdart_model.copy(n_trees=small_xdart_trees)
small_xdart.save(small_xdart_model_file)

small_xdart_effectiveness = float( tree_wise_perf.loc[{'model':xdart_model, 'k':small_xdart_trees, 'metric':ndcg_10}].values )

small_xdart_efficiency = run_condop(small_xdart_model_file, test_dataset_file)

results.loc[len(results)] = [small_xdart.name, small_xdart.n_trees, small_xdart_effectiveness, small_xdart_efficiency]
results

target_model_file = small_xdart_model_file
target_model = small_xdart
target_model.name = "QuickScorer"
target_model_effectiveness = small_xdart_effectiveness

scorer_out = get_ipython().getoutput('{QUICKSCORER}   -d {test_dataset_file}   -m {target_model_file}   -l 64   -r 1   -t 0')
    
print (scorer_out.n)
    
# takes the scoring time in milli-seconds
scoring_time = float(scorer_out.l[-1].split()[-2])* 10**6

results.loc[len(results)] = [target_model.name, target_model.n_trees, target_model_effectiveness, scoring_time]
results

scorer_out = get_ipython().getoutput('{QUICKSCORER}   -d {test_dataset_file}   -m {target_model_file}   -l 64   -r 1   -t 3   -v 8   --avx')

print (scorer_out.n)
    
# takes the scoring time in milli-seconds
scoring_time = float(scorer_out.l[-1].split()[-2])* 10**6

results.loc[len(results)] = ['V-QS', target_model.n_trees, target_model_effectiveness, scoring_time]
results

results["Speed-up"] = results["Scoring Time µs."][0]/ results["Scoring Time µs."]
results



