# 

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
from rnaseq import subset_rmats_junctioncountonly as subset
import pandas as pd
import pybedtools as bt
import glob
import os
from collections import defaultdict
from tqdm import tnrange, tqdm_notebook

all_positive = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.positive.txt')
all_negative = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.negative.txt')

all_jxc = all_positive + all_negative

def get_avg_number_removed(samples, o, e='se'):
    """
    Gets the average number of events removed from a splice event list
    by calculating the number of events before and after duplicate removal.
    
    """
    progress = tnrange(len(samples))
    nums_original = []
    nums_removed = []
    for i in samples:
        # read in original jxc only dataframe (list of events)
        df_before = pd.read_table(i)
        
        # do the subsetting
        df_after = subset.run_subset_rmats_junctioncountonly(i, o, e)
        
        num_events_after = df_after.shape[0]
        num_events_before = df_before.shape[0]
        
        num_events_removed = num_events_before - num_events_after
        
        nums_removed.append(num_events_removed)
        
        if num_events_removed < 0: 
            print("problem", i)
            return 1
        
        progress.update(1)
    return sum(nums_removed) / float(len(nums_removed))

o = '/projects/ps-yeolab3/bay001/tmp/test2.jxc'
get_avg_number_removed(all_jxc, o)

all_sig_positive = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.positive.txt')
all_sig_negative = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.negative.txt')
all_sig = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.significant.txt')
all_original = glob.glob('/projects/ps-yeolab3/bay001/maps/current_annotations/se/*.JunctionCountOnly.txt')

def get_prefix(fn):
    return os.path.basename(fn).split('-SE.MATS')[0]


def build_dictionary_of_files(all_original):
    progress = tnrange(len(all_original))
    d = defaultdict(dict) # dictionary of files
    to_check = [] # list of files that we have no events for
    for fn in all_original:
        prefix = get_prefix(fn)
        original = pd.read_table(fn)
        original = original.shape[0]
        try:
            significant = glob.glob(fn.replace('.txt','.significant.txt'))[0]
            significant = pd.read_table(significant)
            significant = significant.shape[0]
        except IndexError:
            print("{} has no significant events".format(fn))
            significant = 0
            to_check.append(fn)
        try:
            positive = glob.glob(fn.replace('.txt','.positive.txt'))[0]
            positive = pd.read_table(positive)
            positive = positive.shape[0]
        except IndexError:
            positive = 0
        try:
            negative = glob.glob(fn.replace('.txt','.negative.txt'))[0]
            negative = pd.read_table(negative)
            negative = negative.shape[0]
        except IndexError:
            negative = 0
        try:
            positive_collapsed = glob.glob(fn.replace('.txt','.positive.nr.txt'))[0]
            positive_collapsed = pd.read_table(positive_collapsed)
            positive_collapsed = positive_collapsed.shape[0]
        except IndexError:
            positive_collapsed = 0
        try:
            negative_collapsed = glob.glob(fn.replace('.txt','.negative.nr.txt'))[0]
            negative_collapsed = pd.read_table(negative_collapsed)
            negative_collapsed = negative_collapsed.shape[0]
        except IndexError:
            negative_collapsed = 0
        
        d[prefix] = {
            'original_file':fn,
            'original_num':original,
            'significant':significant,
            'significant_positive':positive,
            'significant_negative':negative,
            'significant_positive_collapsed':positive_collapsed,
            'significant_negative_collapsed':negative_collapsed,
        }
        progress.update(1)
        
    return pd.DataFrame(d), to_check

df, to_check = build_dictionary_of_files(all_original)

# check to make sure the files with missing *.significant.txt actually have zero significant events.

def check_missing_sigevents(to_check):
    for fn in to_check:
        df = pd.read_table(fn)
        df = df[(df['IncLevelDifference']>0.05) | (df['IncLevelDifference']<-.05)]
        df = df[(df['PValue'] < 0.05) & (df['FDR'] < 0.1)]
        assert df.shape[0] == 0
check_missing_sigevents(to_check)

df.to_csv('/projects/ps-yeolab3/bay001/gabe_qc_20170612/permanent_data/event_metrics.tsv', sep='\t')

df

dx = pd.read_table('/projects/ps-yeolab3/bay001/maps/current_annotations/se/AARS-BGHLV17-HepG2-SE.MATS.JunctionCountOnly.txt')

dx = dx[(dx['IncLevelDifference']>0.05) | (dx['IncLevelDifference']<-.05)]
dx = dx[(dx['PValue'] < 0.05) & (dx['FDR'] < 0.1)]
dx.shape[0]

df.shape

(df.loc['significant'].sum())/452



