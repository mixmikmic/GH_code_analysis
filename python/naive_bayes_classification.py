get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("poster")

sample_df = pd.read_csv("sample_cases.csv")

# turn issue areas into dummy column
issue_areas = ["criminal procedure","civil rights","first amendment","due process","privacy","attorneys",
              "unions","economic activity","judicial power","federalism","interstate  amendment",
              "federal taxation","miscellaneous","private action"]

for issue, num in zip(issue_areas,range(1,15)):
    sample_df[issue] = sample_df.issueArea.apply(lambda x: 1 if x == num else 0)

# turn decision directions into dummy column (conservative, liberal, neutral)
decision_areas = ["conservative","liberal","neutral"]

for decision, num in zip(decision_areas,range(1,4)):
    sample_df[decision] = sample_df.decisionDirection.apply(lambda x: 1 if x == num else 0)

# function (from lab 9) to vectorize text - adapted to accomodate topics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

def make_xy(df, issue, vectorizer=None):   
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df.text)
    X = X.tocsc()
    y = (df[issue] == 1).values.astype(np.int)
    return X, y

from sklearn.cross_validation import KFold

# function to return cross-validation score (lab 9)
def cv_score(clf, X, y, scorefunc):
    result = 0.
    nfold = 5
    for train, test in KFold(y.size, nfold): # split data into train/test groups, 5 times
        clf.fit(X[train], y[train]) # fit
        result += scorefunc(clf, X[test], y[test]) # evaluate score function on held-out data
    return result / nfold # average

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    not_topic = y == 0
    topic = ~not_topic
    return prob[not_topic, 0].sum() + prob[topic, 1].sum()

"""
Function
--------
calibration_plot
Builds a plot from a classifier and review data

Inputs
-------
clf : Classifier object
    A MultinomialNB classifier
X : (Nexample, Nfeature) array
    The bag-of-words data
Y : (Nexample) integer array
    1 if an opinion is in a certain topic
"""    

def calibration_plot(clf, issue, xtest, ytest):
    prob = clf.predict_proba(xtest)[:, 1]
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))

    #group outcomes into bins of similar probability
    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]
    
    #topic ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
        
    #the calibration plot
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'], color='#7ddbca')
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='#7ddbca')
    plt.ylabel("Empirical P(%s)" % issue)
    
    #the distribution of P(topic)
    ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)
    
    plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
            width=.95 * (bins[1] - bins[0]),
            fc=p[0].get_color(), alpha=0.7, linewidth=0)
    
    plt.xlabel("Predicted P(%s)" % issue)
    plt.ylabel("Number")

# function to run all commands above in sequence
def run_multinb(df, issue):
    
    # create a mask to split data into test and training sets
    itrain, itest = train_test_split(xrange(df.shape[0]), train_size=0.7)
    mask = np.ones(df.shape[0], dtype='int')
    mask[itrain] = 1
    mask[itest] = 0
    mask = (mask == 1)
    
    #the grid of parameters to search over
    alphas = [0, .1, 1, 5, 10, 50]
    min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # find the best value for alpha and min_df, and the best classifier
    best_alpha = None
    best_min_df = None
    maxscore = -np.inf
    for alpha in alphas:
        for min_df in min_dfs:         
            vectorizer = CountVectorizer(min_df = min_df)       
            Xthis, ythis = make_xy(df, issue, vectorizer)
            Xtrainthis = Xthis[mask]
            ytrainthis = ythis[mask]
            clf = MultinomialNB(alpha=alpha)
            cvscore = cv_score(clf, Xtrainthis, ytrainthis, log_likelihood)
            if cvscore > maxscore:
                maxscore = cvscore
                best_alpha, best_min_df = alpha, min_df
    
    print issue.upper()
    print "---------------"
    
    # print and store best alpha values, fit the model on training data on these
    print "alpha: %f" % best_alpha
    print "min_df: %f" % best_min_df
    best_alphas.append(best_alpha)
    best_min_dfs.append(best_min_df)
    vectorizer = CountVectorizer(min_df = best_min_df)
    X, y = make_xy(df, issue, vectorizer)
    xtrain, ytrain, xtest, ytest = X[mask], y[mask], X[~mask], y[~mask]
    clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

    # store and print train and test accuracy
    print "------"
    train_acc.append(clf.score(xtrain, ytrain))
    test_acc.append(clf.score(xtest, ytest))
    print "Accuracy on training data: %0.2f" % (clf.score(xtrain, ytrain))
    print "Accuracy on test data:     %0.2f" % (clf.score(xtest, ytest))
    
    # store and print true/false pos/negs
    tp.append(confusion_matrix(ytest, clf.predict(xtest))[0][0])
    fp.append(confusion_matrix(ytest, clf.predict(xtest))[1][0])
    fn.append(confusion_matrix(ytest, clf.predict(xtest))[0][1])
    tn.append(confusion_matrix(ytest, clf.predict(xtest))[1][1])
    print "------"
    print "True positives:", confusion_matrix(ytest, clf.predict(xtest))[0][0]
    print "False positives:", confusion_matrix(ytest, clf.predict(xtest))[1][0]
    print "False negatives:", confusion_matrix(ytest, clf.predict(xtest))[0][1]
    print "True negatives:", confusion_matrix(ytest, clf.predict(xtest))[1][1]
    
    # get features from results
    words = np.array(vectorizer.get_feature_names())
    x = np.eye(xtest.shape[1])
    probs = clf.predict_log_proba(x)[:, 0]
    ind = np.argsort(probs)

    # top 10 words associated with topic
    good_prob = probs[ind[:10]]
    good_words = words[ind[:10]]
    rep_words.append(good_words)

    print "------"
    print "Words yielding highest P(%s | word)" % issue
    for w, p in zip(good_words, good_prob):
        print "%20s" % w, "%0.2f" % (1 - np.exp(p))
    
    # predict probability of belonging to topic
    prob = clf.predict_proba(X)[:, 0]
    prob_dict[issue] = prob
    predict = clf.predict(X)
    predict_dict[issue] = predict
    
    # print false positives and false negatives
    bad_fp = np.argsort(prob[y == 0])[:5]
    bad_fn = np.argsort(prob[y == 1])[-5:]
    
    print "False positive cases:"
    print '---------------------------'
    for row in bad_fp:
        print df[y == 0].case.irow(row)

    print
    print "False negative cases:"
    print '--------------------------'
    for row in bad_fn:
        print df[y == 1].case.irow(row)
    
    # calibration plot
    calibration_plot(clf, issue, xtest, ytest)

# we can now repeat this process for all 14 topics
issue_areas = ["criminal procedure","civil rights","first amendment","due process","privacy","attorneys",
              "unions","economic activity","judicial power","federalism","interstate  amendment",
              "federal taxation","miscellaneous","private action"]

# data fields to store model outputs
best_alphas,best_min_dfs,train_acc,test_acc,tp,fp,tn,fn,rep_words = [],[],[],[],[],[],[],[],[]
predict_dict = {}
prob_dict = {}

# start with issue 1 (criminal procedure)
run_multinb(sample_df, issue_areas[0])

run_multinb(sample_df, issue_areas[1])

run_multinb(sample_df, issue_areas[2])

run_multinb(sample_df, issue_areas[3])

run_multinb(sample_df, issue_areas[4])

run_multinb(sample_df, issue_areas[5])

run_multinb(sample_df, issue_areas[6])

run_multinb(sample_df, issue_areas[7])

run_multinb(sample_df, issue_areas[8])

run_multinb(sample_df, issue_areas[9])

run_multinb(sample_df, issue_areas[10])

run_multinb(sample_df, issue_areas[11])

run_multinb(sample_df, issue_areas[12])

# make the probabilities into a dataframe
prob_df = pd.DataFrame(prob_dict)
prob_df = prob_df[issue_areas[:-1]]
prob_df.head()

import bottleneck as bn 

# make the probabilities into a matrix, take top threee topics associated with max values
prob_matrix = prob_df.as_matrix()
prob_issues = map(lambda x: list(bn.argpartsort(x,3)[:3]), prob_matrix)

# map indices to issue areas and add results to original sample dataframe
sample_df["predicted_issue_area"] = [[x+1 for x in y] for y in prob_issues] 

# store correct predictions in a new accuracy column
sample_df["predicted_correctly"] = [1 if x in y else 0 for y,x in zip(sample_df["predicted_issue_area"], sample_df["issueArea"])]
print "Accuracy across all issue areas: ", float(float(sum(sample_df["predicted_correctly"]))/float(len(sample_df)))

sample_df[["issueArea","predicted_correctly","predicted_issue_area"]].head()

# create dataframe of output from modeling
model_dict = {}
model_dict["issue"] = issue_areas[:-1]
model_dict["best alpha"] = best_alphas
model_dict["best min_df"] = best_min_dfs
model_dict["training_accuracy"] = train_acc
model_dict["test_accuracy"] = test_acc
model_dict["tp"] = tp
model_dict["tn"] = tn
model_dict["fp"] = fp
model_dict["fn"] = fn
model_dict["words"] = rep_words
model_df = pd.DataFrame(model_dict)

# calculate specificity (true neg rate) and sensitivity (true pos rate)
tpr = [float(float(a) / float(a+b)) for a,b in zip(tp,fn)]
tnr = [float(float(a) / float(a+b)) for a,b in zip(tn,fp)]
model_df["sensitivity"] = tpr
model_df["specificity"] = tnr

# save this csv for comparison for final notebook (on full data set)
model_df.to_csv("naive_bayes_sample_model_results.csv", sep=',', encoding='utf-8',index=False)
model_df.head()

# explore top words associated with each topic area
for wordlist,issue in zip(model_df.words,issue_areas[:-1]):
    print issue.upper() + ": " + ", ".join(wordlist)

# plot training accuracy of each model
plt.figure(figsize=(20,8))
plt.scatter(range(len(model_df.test_accuracy)),model_df.test_accuracy, color="#5db0fd",label="Test Accuracy",s=80,alpha=0.7)
plt.scatter(range(len(model_df.training_accuracy)),model_df.training_accuracy, color="#73e1bd",label="Training Accuracy",s=80,alpha=0.7)
plt.scatter(range(len(model_df.sensitivity)),model_df.sensitivity, color="#ff9060",label="Sensitivity",s=80,alpha=0.7)
plt.scatter(range(len(model_df.specificity)),model_df.specificity, color="#7077e5",label="Specificity",s=80,alpha=0.7)
plt.xticks(np.arange(0,13,1),issue_areas[:-1], rotation=90)
plt.title("Sensitivity, Specificity, and Accuracy of Models by Issue Area")
plt.xlabel("Issue Area")
plt.ylabel("Probabilities")
plt.legend(loc="lower left")
plt.show()

predictions_df = pd.DataFrame(predict_dict)
predictions_df.head()

# plot distribution of all cases by issue type
import collections

issue_dict = {}
for issue in issue_areas[:-1]:
    issue_dict[issue] = sum(predictions_df[issue])

sorted_dict = collections.OrderedDict()
sorted_vals = sorted(issue_dict.values(),reverse=True)
sorted_keys = sorted(issue_dict, key=issue_dict.get,reverse=True)
for key, val in zip(sorted_keys,sorted_vals):
    sorted_dict[key] = val

plt.figure(figsize=(20,12))
plt.grid(zorder=3)
plt.barh(range(len(sorted_dict)),sorted_dict.values(),align='center',color=sns.color_palette("Set2", 14),linewidth=0,zorder=0)
plt.gca().yaxis.grid(False)
plt.gca().invert_yaxis()
plt.yticks(range(len(sorted_dict)),sorted_dict.keys())
plt.xticks(np.arange(0,700,50))
plt.title("Distribution of Syllabi Predictions by Issue Area")
plt.xlabel("Number of Cases")
plt.ylabel("Issue Areas")
plt.show()

predictions_df["num_categories"] = predictions_df.sum(axis=1)

plt.figure(figsize=(20,8))
plt.hist(predictions_df.num_categories,bins=7,alpha=0.7,color="#84e7d1",linewidth=0,zorder=0)
plt.grid(zorder=3)
plt.gca().xaxis.grid(False)
plt.title("Distribution of Number of Topics Assigned per Case")
plt.xlabel("Number of Topics")
plt.ylabel("Number of Cases")
plt.show()

for index in list(predictions_df[predictions_df["num_categories"] >= 5].index):
    print sample_df.iloc[index].case
    print "-----------"
    print sample_df.iloc[index].year
    print issue_areas[sample_df.iloc[index].issueArea-1]
    print 

run_multinb(sample_df, decision_areas[0])

run_multinb(sample_df, decision_areas[1])

run_multinb(sample_df, decision_areas[2])



