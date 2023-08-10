get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd

from rankeval.dataset import Dataset
from rankeval.model import RTEnsemble
from rankeval.analysis.effectiveness import tree_wise_performance

# Global Options

# paths to executable files
QUICKRANK      = "./quickrank/bin/quicklearn"
SCORER         = "./quickrank/bin/quickscore"
QUICKSCORER    = "./QuickScorer/bin/quickscorer"
QUICKSCORER_NS = "./QuickScorer-noscoring/bin/quickscorer"
VPRED          = "./asadi_tkde13/out/VPred"
VPRED_NS       = "./asadi_tkde13-noscoring/out/VPred"
PERF           = "perf"

# paths to Istella-S dataset
train_dataset_file       = "/data/letor-datasets/tiscali/sample/train.txt"
valid_dataset_file       = "/data/letor-datasets/tiscali/sample/vali.txt"
test_dataset_file        = "/data/letor-datasets/tiscali/sample/test.txt"

dataset_size = 681250

# The first row of the test file used by VPred should be: "<# rows of the file> <# features>\n".
vpred_test_dataset_file  = "/data/letor-datasets/tiscali/sample/test.vpred"

# paths to model file
models_folder            = "models"
baseline_model_file      = os.path.join(models_folder, "istella-small.lamdamart.xml")

# setting floating point precision of Pandas
pd.set_option('precision', 1)

# load a QuickRank model
# if no model is available, use the box below to train one!

baseline_model = RTEnsemble(baseline_model_file, name="Baseline", format="QuickRank")

# The code below trains a LambdaMART of 1000 trees.

get_ipython().system('{QUICKRANK}   --algo LAMBDAMART   --num-trees 1000   --shrinkage 0.05   --num-thresholds 0   --num-leaves 64   --min-leaf-support 1   --end-after-rounds 0   --partial 1000   --train {train_dataset_file}   --valid {valid_dataset_file}   --train-metric NDCG   --train-cutoff 10   --model-out ~/quickrank.1000T.64L.xml')

# We use CondOp-based C code as a baseline for the scoring time evaluation

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
    
    # Now running the Conditional Operators scorer by executing the previously compiled C code.
    # QuickScore options:
    #  -h,--help                             print help message
    #  -d,--dataset <arg>                    Input dataset in SVML format
    #  -r,--rounds <arg> (10)                Number of test repetitions
    #  -s,--scores <arg>                     File where scores are saved (Optional).
    print (" 3. Running the compiled model")
    cond_op_scorer_out = get_ipython().getoutput('{SCORER}       -d {dataset_file}       -r {rounds}')
    
    print (cond_op_scorer_out.n)
    
    # takes the scoring time in milli-seconds
    cond_op_scoring_time = float(cond_op_scorer_out.l[-1].split()[-2])* 10**6
    
    return cond_op_scoring_time

condop_efficiency = run_condop(baseline_model_file, test_dataset_file) 

# Store current results
results = pd.DataFrame(columns=['Model', '# Trees', 'Scoring Time µs.'])

results.loc[len(results)] = ['CondOp', baseline_model.n_trees, condop_efficiency]
results

vpred_source = baseline_model_file + ".vpred"

get_ipython().system('{QUICKRANK}   --generator vpred   --model-file {baseline_model_file}   --code-file {vpred_source}')

# Now running the VPred scorer by using the previously converted code.
# note that we are using the original VPred code by Asadi et al. [VPRED].
# The code is available here: https://github.com/lintool/OptTrees

vpred_scorer_out = get_ipython().getoutput('{VPRED}   -ensemble {vpred_source}   -instances {vpred_test_dataset_file}   -maxLeaves 64')
    
print (vpred_scorer_out.n)

# takes the scoring time in milli-seconds
vpred_scoring_time = float(vpred_scorer_out.l[0].split('\t')[1])* 10**6

# Store current results
results.loc[len(results)] = ['VPred', baseline_model.n_trees, vpred_scoring_time]
results

# Now running QuickScorer.
# note that we are using the original QuickScorer code by Lucchese et al. [QS-SIGIR15,QS-TOIS16].
# The code is available under NDA.
#
# Options:
#  -h [ --help ]                     Print help messages.
#  -d [ --dataset ] arg              Path of the dataset to score (SVML format).
#  -r [ --rounds ] arg (=10)         Number of test repetitions.
#  -s [ --scores ] arg               Path of the file where final scores are
#                                    saved.
#  -t [ --tree_type ] arg (=0)       Specify the type of the tree in the
#                                    ensemble:
#                                     - 0 for normal trees,
#                                     - 1 for oblivious trees,
#                                     - 2 for normal trees (reversed blocked),
#                                     - 3 for normal trees (SIMD: SSE/AVX).
#  -m [ --model ] arg                Path of the XML file storing the model.
#  -l [ --nleaves ] arg              Maximum number of leaves in a tree (<= 64).
#  --avx                             Use AVX 256 instructions (at least 8 doc
#                                    blocking).
#  --omp                             Use OpenMP multi-threading document scoring
#                                    (only SIMD: SSE/AVX).

qs_scorer_out = get_ipython().getoutput('{QUICKSCORER}   -d {test_dataset_file}   -m {baseline_model_file}   -l 64   -r 1   -t 0')
    
print (qs_scorer_out.n)
    
# takes the scoring time in milli-seconds
qs_scoring_time = float(qs_scorer_out.l[-1].split()[-2])* 10**6

# Store current results
results.loc[len(results)] = ['QS', baseline_model.n_trees, qs_scoring_time]
results

# Now running Vectorized QuickScorer (AVX2)
# note that we are using the original QuickScorer code by Lucchese et al. [QS-SIGIR16].
# The code is available under NDA.
#
# Options:
#  -h [ --help ]                     Print help messages.
#  -d [ --dataset ] arg              Path of the dataset to score (SVML format).
#  -r [ --rounds ] arg (=10)         Number of test repetitions.
#  -s [ --scores ] arg               Path of the file where final scores are
#                                    saved.
#  -t [ --tree_type ] arg (=0)       Specify the type of the tree in the
#                                    ensemble:
#                                     - 0 for normal trees,
#                                     - 1 for oblivious trees,
#                                     - 2 for normal trees (reversed blocked),
#                                     - 3 for normal trees (SIMD: SSE/AVX).
#  -m [ --model ] arg                Path of the XML file storing the model.
#  -l [ --nleaves ] arg              Maximum number of leaves in a tree (<= 64).
#  -v [ --doc_block_size ] arg (=1)  Document block size (allowed values:
#                                    1,2,4,8,16; 1 means no blocking).
#  --avx                             Use AVX 256 instructions (at least 8 doc
#                                    blocking).
#  --omp                             Use OpenMP multi-threading document scoring
#                                    (only SIMD: SSE/AVX).

vqs_scorer_out = get_ipython().getoutput('{QUICKSCORER}   -d {test_dataset_file}   -m {baseline_model_file}   -l 64   -r 1   -t 3   -v 8   --avx')
    
print (vqs_scorer_out.n)
    
# takes the scoring time in milli-seconds
vqs_scoring_time = float(vqs_scorer_out.l[-1].split()[-2])* 10**6

# Store current results
results.loc[len(results)] = ['v-QS', baseline_model.n_trees, vqs_scoring_time]
results

# Below, perf is used to monitor several behaviours of the scorer:
# - L1 cache performance (references and misses): L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses
# - L3 cache performance (references and misses): cache-references,cache-misses
# - number of instructions and cycles: instructions,cycles
# - total number of branches and branch misprediction: branches,branch-misses

perf_out = get_ipython().getoutput('{PERF} stat -e   L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cycles,cache-references,cache-misses,branches,branch-misses    {QUICKSCORER}       -d {test_dataset_file}       -m {baseline_model_file}       -l 64       -r 1       -t 0')

print (perf_out.n)

# parsing perf output
num_istructions = int(perf_out[20].strip().split(' ')[0].replace(',', ''))
num_cache_ref = int(perf_out[22].strip().split(' ')[0].replace(',', ''))
num_cache_miss = int(perf_out[23].strip().split(' ')[0].replace(',', ''))
num_branches = int(perf_out[24].strip().split(' ')[0].replace(',', ''))
num_branch_misses = int(perf_out[25].strip().split(' ')[0].replace(',', ''))

# Below, perf is used to monitor several behaviours of the scorer:
# - L1 cache performance (references and misses): L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses
# - L3 cache performance (references and misses): cache-references,cache-misses
# - number of instructions and cycles: instructions,cycles
# - total number of branches and branch misprediction: branches,branch-misses

perf_noscoring_out = get_ipython().getoutput('{PERF} stat -e   L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cycles,cache-references,cache-misses,branches,branch-misses    {QUICKSCORER_NS}       -d {test_dataset_file}       -m {baseline_model_file}       -l 64       -r 1       -t 0')
        
print (perf_noscoring_out.n)

# parsing perf output
num_istructions_ns = int(perf_noscoring_out[20].strip().split(' ')[0].replace(',', ''))
num_cache_ref_ns = int(perf_noscoring_out[22].strip().split(' ')[0].replace(',', ''))
num_cache_miss_ns = int(perf_noscoring_out[23].strip().split(' ')[0].replace(',', ''))
num_branches_ns = int(perf_noscoring_out[24].strip().split(' ')[0].replace(',', ''))
num_branch_misses_ns = int(perf_noscoring_out[25].strip().split(' ')[0].replace(',', ''))

# Store current results
perf_results = pd.DataFrame(columns=['Method', 'Instructions', 'Cache Misses', 'Branch Misprediction'])

normalized_instruction_count = (num_istructions - num_istructions_ns) / float(dataset_size * baseline_model.n_trees)
normalized_cache_miss = (num_cache_miss - num_cache_miss_ns) / float(dataset_size * baseline_model.n_trees)
normalized_branch_miss = (num_branch_misses - num_branch_misses_ns) / float(dataset_size * baseline_model.n_trees)

perf_results.loc[len(perf_results)] = ['QS',
                                  normalized_instruction_count,
                                  normalized_cache_miss,
                                  normalized_branch_miss]
perf_results

# Below, perf is used to monitor several behaviours of the scorer:
# - L1 cache performance (references and misses): L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses
# - L3 cache performance (references and misses): cache-references,cache-misses
# - number of instructions and cycles: instructions,cycles
# - total number of branches and branch misprediction: branches,branch-misses

vpred_perf_out = get_ipython().getoutput('{PERF} stat -e   L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cycles,cache-references,cache-misses,branches,branch-misses    {VPRED}       -ensemble {vpred_source}       -instances {vpred_test_dataset_file}       -maxLeaves 64')
        
print (vpred_perf_out.n)

# parsing perf output
num_istructions = int(vpred_perf_out[11].strip().split(' ')[0].replace(',', ''))
num_cache_ref = int(vpred_perf_out[13].strip().split(' ')[0].replace(',', ''))
num_cache_miss = int(vpred_perf_out[14].strip().split(' ')[0].replace(',', ''))
num_branches = int(vpred_perf_out[15].strip().split(' ')[0].replace(',', ''))
num_branch_misses = int(vpred_perf_out[16].strip().split(' ')[0].replace(',', ''))

# Below, perf is used to monitor several behaviours of the scorer:
# - L1 cache performance (references and misses): L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses
# - L3 cache performance (references and misses): cache-references,cache-misses
# - number of instructions and cycles: instructions,cycles
# - total number of branches and branch misprediction: branches,branch-misses

vpred_perf_noscoring_out = get_ipython().getoutput('{PERF} stat -e   L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cycles,cache-references,cache-misses,branches,branch-misses    {VPRED_NS}       -ensemble {vpred_source}       -instances {vpred_test_dataset_file}       -maxLeaves 64')
        
print (vpred_perf_noscoring_out.n)

# parsing perf output
num_istructions_ns = int(vpred_perf_noscoring_out[11].strip().split(' ')[0].replace(',', ''))
num_cache_ref_ns = int(vpred_perf_noscoring_out[13].strip().split(' ')[0].replace(',', ''))
num_cache_miss_ns = int(vpred_perf_noscoring_out[14].strip().split(' ')[0].replace(',', ''))
num_branches_ns = int(vpred_perf_noscoring_out[15].strip().split(' ')[0].replace(',', ''))
num_branch_misses_ns = int(vpred_perf_noscoring_out[16].strip().split(' ')[0].replace(',', ''))

# Store current results
normalized_instruction_count = (num_istructions - num_istructions_ns) / float(dataset_size * baseline_model.n_trees)
normalized_cache_miss = (num_cache_miss - num_cache_miss_ns) / float(dataset_size * baseline_model.n_trees)
normalized_branch_miss = (num_branch_misses - num_branch_misses_ns) / float(dataset_size * baseline_model.n_trees)

perf_results.loc[len(perf_results)] = ['VPred',
                                  normalized_instruction_count,
                                  normalized_cache_miss,
                                  normalized_branch_miss]
perf_results

# setting up environment variables for OpenMP
os.environ['OMP_NUM_THREADS']='32'
os.environ['OMP_DISPLAY_ENV']='VERBOSE'
os.environ['OMP_SCHEDULE']='auto'
os.environ['GOMP_CPU_AFFINITY']='0-7,8-15'

# Now running Multi-threaded Vectorized QuickScorer.
# Options:
#  -h [ --help ]                     Print help messages.
#  -d [ --dataset ] arg              Path of the dataset to score (SVML format).
#  -r [ --rounds ] arg (=10)         Number of test repetitions.
#  -s [ --scores ] arg               Path of the file where final scores are
#                                    saved.
#  -t [ --tree_type ] arg (=0)       Specify the type of the tree in the
#                                    ensemble:
#                                     - 0 for normal trees,
#                                     - 1 for oblivious trees,
#                                     - 2 for normal trees (reversed blocked),
#                                     - 3 for normal trees (SIMD: SSE/AVX).
#  -m [ --model ] arg                Path of the XML file storing the model.
#  -l [ --nleaves ] arg              Maximum number of leaves in a tree (<= 64).
#  -v [ --doc_block_size ] arg (=1)  Document block size (allowed values:
#                                    1,2,4,8,16; 1 means no blocking).
#  --avx                             Use AVX 256 instructions (at least 8 doc
#                                    blocking).
#  --omp                             Use OpenMP multi-threading document scoring
#                                    (only SIMD: SSE/AVX).

scorer_out = get_ipython().getoutput('{QUICKSCORER}   -d {test_dataset_file}   -m {baseline_model_file}   -l 64   -r 1   -t 3   -v 8   --avx   --omp')
    
print (scorer_out.n)
    
# takes the scoring time in milli-seconds
scoring_time = float(scorer_out.l[-1].split()[-2])* 10**6

# Store current results
results.loc[len(results)] = ['vQS-OMP', baseline_model.n_trees, scoring_time]
results

from rankeval.dataset.datasets_fetcher import load_dataset

dataset_container = load_dataset(dataset_name='istella-sample',
                                download_if_missing=True, 
                                force_download=False, 
                                with_models=False)

# We now use RankEval to score the test file.
scorer_out = get_ipython().run_line_magic('timeit', '-o baseline_model.score(dataset_container.test_dataset, False)')

