# Imports and setup.
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter, defaultdict
from itertools import chain, combinations, cycle
from IPython.display import display
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import tarfile
from urllib.request import urlretrieve
path = 'data/'
os.environ['NOBULL_PATH'] = path
import u
get_ipython().run_line_magic('matplotlib', 'inline')
u.config_matplotlib()

def download_data(path):
    """
    Download any required files if not already present.
    """
    url = 'https://www.dropbox.com/s/5lvcowbq9kqpvkc/data.tgz?dl=1'
    if not os.path.exists(path + os.path.sep + 'model.w2v'):
        zipname = 'data.tgz'
        print('fetching data (1.5G)')
        urlretrieve(url, zipname)
        tar = tarfile.open(zipname, "r:gz")
        print('extracting %s' % zipname)
        tar.extractall()
        tar.close()
    else:
        print('data already exists in %s' % path)

download_data(path)

# Read raw data.
task1_posts = u.load_posts(path + 'task1_data.json', path + 'task1_labels.json')
print('read %d posts' % len(task1_posts))

def get_feature_indices(feature_names, feature_classes):
    print('%d feature classes, %d features' % (len(feature_classes), sum(len(f) for f in feature_names)))
    res = {}
    i = 0
    for fc, fns in zip(feature_classes, feature_names):
        res[fc] = np.arange(i, i+len(fns))
        i += len(fns)
    return res
    
def concat_indices(feature_class2indices, classes):
    return np.concatenate([feature_class2indices[c] for c in classes])
    
def enum_feature_subsets(feature_class2indices):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(feature_class2indices.keys())
    return ((classes, concat_indices(feature_class2indices, classes))
             for classes in chain.from_iterable(combinations(s, r)
                                                for r in range(1, len(s)+1)))

def select_results(results, num_features, used_features):
    selected = []
    for i, r in enumerate(results.iterrows()):
        if len(r[1]['features']) == num_features and len(used_features - set(r[1]['features'])) == 0:
            selected.append(i)
    return results.iloc[selected]

def print_results_table(results):
    """
    Print feature comparison table.
    """
    res = select_results(results, 2, set(['Unigram']))
    table = pd.concat([results[results.features==('Unigram',)],
                       res.sort_values('AUC', ascending=True),
                       results.sort_values('AUC', ascending=False).head(1)])
    print('best features:', table.features.values[-1])
    pd.options.display.max_colwidth = 100
    names = [t[0] if len(t) == 1 else 'U + %s' % t[1] for t in table['features']]
    names[-1] = 'Best'
    table['features'] = names
    table = table[['features', 'AUC', 'F1', 'Precision', 'Recall']]
    table = table.set_index('features')
    display(table.iloc[0])
    display(table)
    print(table.to_latex(bold_rows=True, float_format='%.3f', index=True))
    
def get_lines():
    random.seed(42)
    markers = ['o', '^', 's', '*', 'D']
    lines = ['-', '--', '-.', ':', '--']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(5):
        mi = list(markers)
        random.shuffle(mi)
        markers += mi
        li = list(lines)
        random.shuffle(li)
        lines += li
        ci = list(colors)
        random.shuffle(ci)
        colors += ci
    return cycle(['%s%s%s' % (c, m, l) for c, l, m in zip(colors, lines, markers)])

def task1_expts(posts, lead_time=3, max_lead_time=10):
    """
    Perform all experiments for Task 1 and generate Table 2.
    """
    task1_sampled_posts = u.sample_posts_task1(posts, lead_time=max_lead_time*60*60)
    task1_sampled_posts = u.set_observed_comments(task1_sampled_posts, lead_time=lead_time*60*60)
    X, vec, feature_names, feature_classes = u.vectorize(task1_sampled_posts)
    feature_class2indices = get_feature_indices(feature_names, feature_classes)
    display(sorted([(fc, len(i)) for fc, i in feature_class2indices.items()], key=lambda x: x[1]))
    all_results = []
    y = np.array([1 if p['num_hostile'] > 0 else 0 for p in task1_sampled_posts])
    label_counts = Counter(y)
    print(label_counts)
    for feature_classes, col_indices in enum_feature_subsets(feature_class2indices):                        
        res = {
                'features': feature_classes,
                'n_instances': len(y),
                'n_pos': label_counts[1],
                'n_neg': label_counts[0]}
        XX = X[:,col_indices]
        # run cross-validation
        n_comments = np.array([p['n_comments_observed'] for p in task1_sampled_posts])
        res.update(u.cv(XX, y, n_splits=10, n_comments=n_comments))
        all_results.append(res)
    return pd.DataFrame(all_results), feature_class2indices
            
task1_results, feature_class2indices = task1_expts(task1_posts, lead_time=3)
print_results_table(task1_results)

# Create table of AUC vs lead time
def task1_lead_time_fig(posts, lead_times, features):
    """
    Produce Figure 3, hostility presence forecasting accuracy
    as lead time increases.
    """
    # resample and vectorize according to each lead_time
    all_results = []
    # to ensure comparability, we'll sample posts using longest lead times, then
    # reuse the for shorter lead times
    posts = u.sample_posts_task1(posts, lead_time=max(lead_times)*60*60)
    for lead_time in lead_times:
        task1_sampled_posts = u.set_observed_comments(posts, lead_time=lead_time*60*60)
        X, vec, feature_names, feature_classes = u.vectorize(task1_sampled_posts)
        feature_class2indices = get_feature_indices(feature_names, feature_classes)
        y = np.array([1 if p['num_hostile'] > 0 else 0 for p in task1_sampled_posts])        
        label_counts = Counter(y)
        n_comments = np.array([p['n_comments_observed'] for p in task1_sampled_posts])
        for feature_list in features:
            col_indices = concat_indices(feature_class2indices, feature_list)
            XX = X[:, col_indices]
            res = {
                    'features': feature_list,
                    'lead_time': lead_time,
                    'n_instances': len(y),
                    'n_pos': label_counts[1],
                    'n_neg': label_counts[0]}            
            # run cross-validation
            res.update(u.cv(XX, y, n_splits=10, n_comments=n_comments))
            all_results.append(res)
    results_df = pd.DataFrame(all_results)
    plot_task1_lead_time_fig(results_df)
    return results_df
            
def plot_task1_lead_time_fig(task1_res_fig, nfolds=10):
    plt.figure()
    linecycler = get_lines()
    for fnames in sorted([x for x in set(task1_res_fig.features)], key=lambda x: -len(x)):
        df = task1_res_fig[task1_res_fig.features==fnames]
        rr = df.sort_values('lead_time')[['lead_time', 'AUC', 'AUC_sd']].values
        xvals = rr[:,0]
        yvals = rr[:,1]
        stderrs = rr[:,2] / math.sqrt(nfolds) # standard error
        marker = next(linecycler)
        plt.plot(xvals, yvals, marker, label='+'.join(fnames))
        plt.errorbar(xvals, yvals, fmt=marker, yerr=stderrs)
    plt.xlabel('lead time (hours)')
    plt.ylabel('AUC')
    plt.legend(loc='lower left')
    plt.setp(plt.legend().get_texts(), fontsize='12') 
    plt.ylim((.73, .855))
    plt.tight_layout()
    plt.savefig('forecast_time.pdf')
    plt.show()

task1_lead_time_results = task1_lead_time_fig(task1_posts,
                                              lead_times=[1, 3, 5, 8, 10],
                                              features=[('Unigram', 'lex'),
                                                        ('Unigram', 'lex', 'w2v'),
                                                        ('Unigram', 'lex', 'n-w2v'),
                                                        ('Unigram', 'lex', 'n-w2v', 'prev-post', 'trend'),
                                                        ]
                                             )

def task1_n_comments_fig(posts, features, lead_time=3):
    """
    Produce Figure 4, hostility presence forecasting AUC as the
    number of observed comments increases.
    """
    task1_sampled_posts = u.sample_posts_task1(posts,
                                               lead_time=lead_time*60*60)
    X, vec, feature_names, feature_classes = u.vectorize(task1_sampled_posts)
    feature_class2indices = get_feature_indices(feature_names, feature_classes)
    y = np.array([1 if p['num_hostile'] > 0 else 0 for p in task1_sampled_posts])        
    label_counts = Counter(y)
    all_results = []
    n_comments = np.array([p['n_comments_observed'] for p in task1_sampled_posts])
    for feature_list in features:
        col_indices = concat_indices(feature_class2indices, feature_list)
        XX = X[:, col_indices]
        res = {
                'features': feature_list,
                'lead_time': lead_time,
                'n_instances': len(y),
                'n_pos': label_counts[1],
                'n_neg': label_counts[0]}            
        res.update(u.cv(XX, y, n_splits=10, n_comments=n_comments))
        all_results.append(res)                                             
    df = pd.DataFrame(all_results)
    plot_task1_n_comments_fig(df)
    return df

def plot_task1_n_comments_fig(df):
    bins = {'1': [1],
            '2': [2],
            '3': [3],
            '4-6': [4,5,6],
            '7-9': [7,8,9],
            '>=10': range(10,500)
           }

    plt.figure()
    linecycler = get_lines()
    for features, by_n_comments in df[['features', 'by_n_comments']][::-1].values:
        # group results by number of comments
        nc2res = defaultdict(list)
        for x in by_n_comments:
            nc2res[x[2]].append((x[0], x[1]))
        rocs = []
        for label, ncs in sorted(bins.items()):
            res = []
            for nc in ncs:
                res.extend(nc2res[nc])
            rocs.append(roc_auc_score([v[0] for v in res],
                                      [v[1] for v in res], average=None))
        plt.plot(rocs, next(linecycler), label='+'.join(features))
    plt.xticks(range(len(bins)), sorted(bins))
    plt.legend(loc='lower right')
    plt.xlabel('number of observed comments')
    plt.ylabel('AUC')
    plt.ylim((.4, 1))
    plt.setp(plt.legend().get_texts(), fontsize='12') 
    plt.savefig('forecast_comments.pdf')
    plt.show()


n_comments_res = task1_n_comments_fig(task1_posts,
                                      features=[('Unigram', 'lex'),
                                                ('Unigram', 'lex', 'w2v'),
                                                ('Unigram', 'lex', 'n-w2v'),
                                                ('Unigram', 'lex', 'n-w2v', 'prev-post', 'trend'),
                                                ],
                                      lead_time=3)

"""
Print the top features per class according to the logistic
regression coefficients, including the top terms in each word2vec dimension.
"""

def load_predicted_vectors(words, w2v=u.w2v_model_3gram, dim=100):
    vecs = []
    for wd in words:
        vecs.append(u.get_vector(wd, w2v, dim))
    return np.array(vecs)

def get_top_w2v_words(words, word_vecs, idx, n=20):
    return words[word_vecs[:,idx].argsort()[::-1][:n]]

def get_top_features_task1(task1_posts, lead_time=3, nfeats=40,
                             features=('Unigram', 'user', 'trend', 'lex',
                                      'n-w2v', 'final-com', 'prev-post')):
    task1_sampled_posts = u.sample_posts_task1(task1_posts, lead_time=3*60*60)
    X, vec, feature_names, feature_classes = u.vectorize(task1_sampled_posts)
    feature_class2indices = get_feature_indices(feature_names, feature_classes)
    y = np.array([1 if p['num_hostile'] > 0 else 0 for p in task1_sampled_posts])
    col_indices = concat_indices(feature_class2indices, features)
    XX = X[:,col_indices]
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(XX, y)
    fnames = np.concatenate(feature_names)
    fnames = fnames[col_indices]
    words = np.array(feature_names[feature_classes.index('Unigram')])
    word_vecs = load_predicted_vectors(words)
    results = []
    for i in np.argsort(clf.coef_[0])[::-1][:nfeats]:
        res = {'feature': fnames[i], 'coef': clf.coef_[0][i]}
        if 'neww2v' in fnames[i]:
            idx = int(re.findall('_([0-9]+)\-', fnames[i])[0])
            res['w2v'] = ' '.join(get_top_w2v_words(words, word_vecs, idx))
        results.append(res)
    results = pd.DataFrame(results)
    display(results)
    return results

get_top_features_task1(task1_posts)

task2_posts = u.load_posts(path + 'task2_data.json', path + 'task2_labels.json')
u.set_n_comments_observed_task2(task2_posts)
print('read %d posts' % len(task2_posts))

# Vectorize
X, vec, feature_names, feature_classes = u.vectorize(task2_posts)
X.shape

feature_class2indices = get_feature_indices(feature_names, feature_classes)
sorted([(fc, len(i)) for fc, i in feature_class2indices.items()], key=lambda x: x[1])

def task2_expts(posts, X, vec, feature_class2indices,
                max_for_neg_class=1, min_for_pos_class=10):
    """
    Perform task 2 experiments and produce Table 3, 
    forecasting AUC with N=10
    """
    all_results = []
    idx = u.filter_by_num_hostile(posts,
                                  max_for_neg_class=max_for_neg_class,
                                  min_for_pos_class=min_for_pos_class)
    Xi = X[idx]
    postsi = posts[idx]
    y = np.array([1 if p['num_hostile'] >= min_for_pos_class else 0
                  for p in postsi])
    label_counts = Counter(y)
    for feature_classes, col_indices in enum_feature_subsets(feature_class2indices):                        
        res = {
                'features': feature_classes,
                'max_for_neg_class': max_for_neg_class,
                'min_for_pos_class': min_for_pos_class,
                'n_instances': len(idx),
                'n_pos': label_counts[1],
                'n_neg': label_counts[0]}
        XX = Xi[:,col_indices]
        # run cross-validation
        res.update(u.cv(XX, y, n_splits=10))
        all_results.append(res)
    return pd.DataFrame(all_results)
    
task2_results = task2_expts(task2_posts, X, vec, feature_class2indices)
print_results_table(task2_results)

"""
Produce Figure 5, hostility intensity forecasting AUC as the
positive class threshold increases.
"""
def task2_min_for_pos_class_fig(task2_results, posts, X, vec, feature_class2indices,
                                features,
                                max_for_neg_class=1,
                                min_for_pos_class_list=range(5,11)):
    col_indices = concat_indices(feature_class2indices, features)
    X = X[:,col_indices]
    all_results = []
    for min_for_pos_class in min_for_pos_class_list:
        idx = u.filter_by_num_hostile(posts,
                                      max_for_neg_class=max_for_neg_class,
                                      min_for_pos_class=min_for_pos_class)
        Xi = X[idx]
        postsi = posts[idx]
        y = np.array([1 if p['num_hostile'] >= min_for_pos_class else 0
                      for p in postsi])
        
        label_counts = Counter(y)
        res = {
                'features': feature_classes,
                'max_for_neg_class': max_for_neg_class,
                'min_for_pos_class': min_for_pos_class,
                'n_instances': len(idx),
                'n_pos': label_counts[1],
                'n_neg': label_counts[0]}
            
        # run cross-validation
        res.update(u.cv(Xi, y, n_splits=10))
        all_results.append(res)
    results_df = pd.DataFrame(all_results)
    plot_task2_fig(results_df)
    return results_df
            
def plot_task2_fig(task2_res_fig, nfolds=10):
    rr = task2_res_fig.sort_values('min_for_pos_class')[['min_for_pos_class', 'AUC', 'AUC_sd']].values
    xvals = rr[:,0]
    yvals = rr[:,1]
    stderrs = rr[:,2] / math.sqrt(nfolds) ## assuming 10-fold cv
    plt.figure()
    plt.plot(xvals, yvals, 'bo-')
    plt.errorbar(xvals, yvals, fmt='b', yerr=stderrs)
    plt.xlabel('minimum number of hostile comments\nin positive class')
    plt.ylabel('AUC')
    plt.tight_layout()
    plt.savefig('intensity_threshold.pdf')
    plt.show()

    
task2_pos_class_res = task2_min_for_pos_class_fig(task2_results, task2_posts, X, vec, feature_class2indices,
                            features=('Unigram', 'lex', 'n-w2v', 'prev-post', 'trend', 'user', 'final-com'),
                            max_for_neg_class=1,
                            min_for_pos_class_list=range(5,16))

"""
Print the top features for task 2 according to the logistic regression coefficients.
"""
def get_top_features_task2(task2_posts,
                             X, vec, feature_class2indices,
                             feature_names,
                             feature_classes,
                             features=('Unigram', 'user', 'trend', 'lex',
                                       'n-w2v', 'final-com', 'prev-post'),
                             min_for_pos_class=10,
                             max_for_neg_class=1,
                             nfeats=40):
    idx = u.filter_by_num_hostile(task2_posts,
                                  max_for_neg_class=max_for_neg_class,
                                  min_for_pos_class=min_for_pos_class)
    Xi = X[idx]
    postsi = task2_posts[idx]
    y = np.array([1 if p['num_hostile'] >= min_for_pos_class else 0
                  for p in postsi])
    col_indices = concat_indices(feature_class2indices, features)
    XX = Xi[:,col_indices]
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(XX, y)
    fnames = np.concatenate(feature_names)
    fnames = fnames[col_indices]
    words = np.array(feature_names[feature_classes.index('Unigram')])
    word_vecs = load_predicted_vectors(words)
    results = []
    for i in np.argsort(clf.coef_[0])[::-1][:nfeats]:
        res = {'feature': fnames[i], 'coef': clf.coef_[0][i]}
        if 'neww2v' in fnames[i]:
            idx = int(re.findall('_([0-9]+)\-', fnames[i])[0])
            res['w2v'] = ' '.join(get_top_w2v_words(words, word_vecs, idx))
        results.append(res)
    results = pd.DataFrame(results)
    return results

get_top_features_task2(task2_posts,
                         X, vec, feature_class2indices,
                         feature_names,
                         feature_classes,
                         features=('Unigram', 'user', 'trend', 'lex',
                                   'n-w2v', 'final-com', 'prev-post'),
                         min_for_pos_class=10,
                         max_for_neg_class=1)

"""
Print top terms for task1 and task2 according to chi-squared.
"""
def top_hostile_terms(task1_posts, min_for_pos_class=10):
    comments_task1 = []
    comments_task2 = []
    labels_task1 = []
    labels_task2 = []
    for p in task1_posts:
        task2_label = 1 if p['num_hostile'] >= min_for_pos_class else 0
        for c,l in zip(p['comments'], p['labels']):
            labels_task1.append(0 if l=='Innocuous' else 1)
            text = u.cleanText(c)
            comments_task1.append(text)
            if labels_task1[-1] == 1:
                comments_task2.append(text)
                labels_task2.append(task2_label)
    vec = CountVectorizer(min_df=5, binary=True)
    X1 = vec.fit_transform(comments_task1)
    y1 = np.array(labels_task1)
    feats1 = np.array(vec.get_feature_names())
    X2 = vec.fit_transform(comments_task2)
    y2 = np.array(labels_task2)
    feats2 = np.array(vec.get_feature_names())
    
    def top_coef(X, y, feats, n=50):
        chi, _ = chi2(X, y)
        pos_counts = X[np.where(y==1)].sum(axis=0).A1
        neg_counts = X[np.where(y==0)].sum(axis=0).A1
        clf = LogisticRegression()
        clf.fit(X,y)
        coef = clf.coef_[0]
        for i in np.argsort(chi)[::-1][:n]:
            if coef[i] > 0:
                print(chi[i], pos_counts[i], neg_counts[i], feats[i])
    print('top terms predictive of hostile vs. non-hostile comment')
    top_coef(X1, y1, feats1, n=50)
    print('\n\n\ntop terms predictive of intense vs. non-intense hostility')
    top_coef(X2, y2, feats2, n=50)

top_hostile_terms(task1_posts, min_for_pos_class=10)

