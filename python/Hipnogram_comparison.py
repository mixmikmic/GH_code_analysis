import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict 
import parse_hipnogram as ph
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import sklearn.metrics as metrics
from matplotlib.dates import  DateFormatter
import seaborn as sns
from datetime import timedelta
plt.rcParams['figure.figsize'] = (9.0, 5.0)
get_ipython().magic('matplotlib notebook')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

plt.style.use('ggplot')

from statsmodels.stats.inter_rater import cohens_kappa, to_table
labels = ['rem','N1', 'N2', 'N3','wake']

stage_to_num = {'W':5, 'R':1, 'N1':2 , 'N2':3, 'N3':4 }
dummy_dict = {'stage_1': 'rem', 'stage_2' : 'N1', 'stage_3' : 'N2', 'stage_4': 'N3', 'stage_5':'wake'}
stage_color_dict = {'N1' : 'royalblue', 'N2' :'forestgreen', 'N3' : 'coral', 'rem' : 'plum', 'wake' : 'lightgray' }

original_accuracy_dict = {}

def prepare_hipnograms(night):
    # load hipnograms, resample them and select their common time window

    psg_hipnogram = ph.parse_psg_stages(night = night).resample('1s').fillna(method = 'ffill')

    # Shift the neuroon hipnogram by the offset identified in eeg signals cross correaltion
    neuroon_hipnogram = ph.parse_neuroon_stages(night = night, time_shift = -160)
    neuroon_hipnogram = neuroon_hipnogram.resample('1s').fillna(method = 'ffill')

    # Trim hipnograms to the common time window so the confusion matrix calculations are accurate
    neuroon_hipnogram = neuroon_hipnogram.between_time('23:00', '06:00')

    psg_hipnogram = psg_hipnogram.between_time('23:00', '06:00')
    
    return neuroon_hipnogram, psg_hipnogram 

def plot_main_confusion_matrix(night):
    
    neuroon_hipnogram, psg_hipnogram = prepare_hipnograms(night)

    # Create a list of staging predictions
    true_stage = psg_hipnogram['stage_num'].as_matrix()
    predicted_stage = neuroon_hipnogram['stage_num'].astype(int).as_matrix()

    fig, axes = plt.subplots()
    fig.suptitle('all stages confusion matrix')

    # Compute the confusion matrix. The values in the cells are minutes (hipnograms are resampled to 1 hz resolution, thus dividing by 60 produces minutes)
    cm = confusion_matrix(true_stage, predicted_stage) / 60.0

    sns.heatmap(cm, annot = True, xticklabels =labels,  yticklabels = labels, fmt = '.1f', ax = axes, vmax = 90)

    axes.set_ylabel('psg')
    axes.set_xlabel('neuroon')

    # Compute precision, recall and f1-score for the multilabel classification from neuroon
    report = metrics.classification_report(true_stage,predicted_stage, target_names = labels )
    print(report)

    # Compute accuracy, give it this name to use after permutation
    original_accuracy = metrics.accuracy_score(true_stage,predicted_stage,)
    print('accuracy: %.2f'%original_accuracy)
    original_accuracy_dict[night] = original_accuracy 
    waking_performance(cm)

def waking_performance(cm):
    
    rem = 0
    n1 = 1
    n2 = 2
    n3 = 3
    wake =4
    
    # Correct waking: Sum of TP for shallow sleep stages, i.e. both neuroon and psg said its time to wake up
    correct_waking = cm[n1, n1].sum() + cm[n2, n2].sum()  + cm[wake, wake].sum()
    
    # Missed waking: Sum where psg said it was shallow sleep but neuroon said it was deep, i.e. psg said its time to wake up, neuroon said its better to sleep 
    missed_waking = cm[[n1, n2 , wake], rem].sum() +   cm[[n1, n2 , wake], n3].sum()
    
    # Sum of false negatives for deep sleep, i.e. psg said its better to sleep, neuroon said its time to wake up
    incorrect_waking = cm[rem, [n1, n2, wake]].sum() + cm[n3, [n1, n2, wake]].sum()
    
    # Both psg and neuroon let the subject sleep
    correct_sleeping = cm[rem, rem].sum() + cm[n3, n3].sum()
    
    # combine waking decisions into a confusion matrix
    waking_matrix = np.array([[correct_waking, missed_waking],[incorrect_waking, correct_sleeping]])
    
    fig, axes = plt.subplots()
    sns.heatmap(waking_matrix, annot = True, xticklabels =['yes', 'no'],  yticklabels = ['yes', 'no'], fmt = '.1f', ax = axes)
    
    axes.set_ylabel('psg')    
    axes.set_xlabel('neuroon')
    fig.suptitle('Czy mozna wybudzac?')

plot_main_confusion_matrix(1)

plot_main_confusion_matrix(2)

def binarize_stages(hipnogram):
    return pd.get_dummies(hipnogram, prefix = 'stage')


def confusion_matrix_separate(night):
    
    neuroon_hipnogram, psg_hipnogram = prepare_hipnograms(night)

    
    # make a confusion matrix for each stage binarized (stage = 1, all_other = 0)
    # TODO, neuroon stages saved as float, change to int upstream
    neuroon_binarized = binarize_stages(neuroon_hipnogram['stage_num'].astype(int))
    psg_binarized = binarize_stages(psg_hipnogram['stage_num'])
    
    fig_p, axes_p = plt.subplots(nrows = 2,ncols = 2, figsize = (8,8))
    fig_n, axes_n = plt.subplots(nrows = 2,ncols = 2, figsize = (8,8))
    
    fig_p.suptitle('Confusion matrixes normalized to psg (column sum)')
    fig_n.suptitle('Confusion matrixes normalized to neuroon (row sum)')
    
    roc_fig, roc_axes = plt.subplots(figsize = (6,6))
    roc_fig.suptitle('ROC')
        
    for neuroon_stage, axp, axn in zip(neuroon_binarized, axes_p.reshape(-1), axes_n.reshape(-1)):
        
        stage_predicted = neuroon_binarized[neuroon_stage].as_matrix()
        stage_true = psg_binarized[neuroon_stage].as_matrix()

        # Compute the confusion matrix [[tp, fn], [fp, tn]]
        cm = confusion_matrix(stage_true, stage_predicted)
        
        
        # Confusion matrix is organized accroding to dummy coding, so the upper right cell [0,0] will have true negatives (psg = 0 and neuroon = 0)
        # This is not accroding to the conventional way of viualizing the matrix, so we'll rotate it
        cm = np.rot90(cm, 2)
        
        # Compute the cohen kappa for the confusion matrix
        table = cohens_kappa(cm)

        # Normalize the confusion matrix by row (i.e by the total length for stage detected by psg)
        cm_normalized_p = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Normalize the confusion matrix by row (i.e by the total length for stage detected by neuroon)
        cm_normalized_n = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        
        # Make a plot of scores in the roc space, i.e. fp vs tp
        roc_axes.plot(cm_normalized_p[1,0], cm_normalized_p[0,0], marker= 'o', color = stage_color_dict[dummy_dict[neuroon_stage]], label = dummy_dict[neuroon_stage])
        
        #Plot the whole confusion matrix normalized to show percentages
        sns.heatmap(cm_normalized_p, annot = True, xticklabels = ['yes','no'],                    yticklabels = ['yes','no'],fmt = '.2f', ax = axp, linewidths=.5,                   vmin = 0.0, vmax = 1.0)
        sns.heatmap(cm_normalized_n, annot = True, xticklabels = ['yes','no'],                    yticklabels = ['yes','no'],fmt = '.2f', ax = axn, linewidths=.5,                   vmin = 0.0, vmax = 1.0)
        
        axp.set_title(dummy_dict[neuroon_stage] + '\n Cohen\'s kappa: %.2f / %.2f'%(table.kappa, table.kappa_max))
        axn.set_title(dummy_dict[neuroon_stage] + '\n Cohen\'s kappa: %.2f / %.2f'%(table.kappa, table.kappa_max))

    axes_p[0,0].set_ylabel('psg')
    axes_p[1,0].set_ylabel('psg')
    axes_p[1,0].set_xlabel('neuroon')
    axes_p[1,1].set_xlabel('neuroon')

        #print(dummy_dict[neuroon_stage])
        # uncomment to see other results, confidence intervals 
        # print('max kappa %.2f'%table.kappa_max)
        #print(table)
        
    roc_axes.plot([0,1], [0,1], color = 'black', linestyle = '--', alpha = 0.5)
    roc_axes.set_xlabel('false_positive')
    roc_axes.set_ylabel('true_positive')
    roc_axes.legend(loc = 'best')
    
    fig_p.tight_layout()
    fig_n.tight_layout()

confusion_matrix_separate(1)

confusion_matrix_separate(2)

# Permutation test by shuffling (without replacement) the stage labels neuroon assigned in original 30sec freq staging
def permute_neuroon_staging(night):
    # load hipnograms, resample them and select their common time window
    neuroon_hipnogram = ph.parse_neuroon_stages(time_shift = -160, night = night, permute = True).resample('1s').fillna(method = 'ffill')
    psg_hipnogram = ph.parse_psg_stages(night = night).resample('1s').fillna(method = 'ffill')

    # Get the start and end of the time window covered by both hipnograms
    start = neuroon_hipnogram.index.searchsorted(psg_hipnogram.index.get_values()[0])
    end = psg_hipnogram.index.searchsorted(neuroon_hipnogram.index.get_values()[-1])

    # Trim hipnograms to the common time window so the confusion matrix calculations are accurate
    # +1 and -1 because events got cut in half, resulting in ends without starts
    neuroon_hipnogram = neuroon_hipnogram.ix[start ::]
    # +1 because upper bound is not included
    psg_hipnogram = psg_hipnogram.ix[0:end +1]
    
    true_stage = psg_hipnogram['stage_num'].as_matrix()
    predicted_stage = neuroon_hipnogram['stage_num'].astype(int).as_matrix()
    
    # Measure accuracy score 
    acc_score = metrics.accuracy_score(true_stage,predicted_stage)
    
    return acc_score


def run_perm(night):

    # run permutation test
    num_perm = 100
    permuted_accuracy = []
    for i in range(num_perm):
        permuted_accuracy.append(permute_neuroon_staging(night))

    permuted_accuracy = np.array(permuted_accuracy)
    
    fig, axes = plt.subplots(figsize = (8,6))
    axes.axvline(original_accuracy_dict[night], color = 'k', linestyle = '--', label = 'original score')
    sns.distplot(permuted_accuracy, ax = axes, label = 'permutation scores')

    axes.set_title('accuracy permutation test')
    axes.set_ylabel('number of occurences')
    axes.set_xlabel('accuracy score')
    axes.legend()

run_perm(1)

run_perm(2)

