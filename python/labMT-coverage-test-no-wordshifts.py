from os import listdir
from os.path import isfile, join
from json import loads
from re import findall,UNICODE
import sys
sys.path.append("/Users/andyreagan/tools/python")
from kitchentable.dogtoys import *
from labMTsimple.labMTsimple.speedy import LabMT
my_labMT = LabMT()
from labMTsimple.labMTsimple.storyLab import *
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from analyzeAll import loadMovieReviews

labMT_center = my_labMT.scorelist.mean()

my_labMT.data["happy"]

# want to compute the average rank, so we can remove words
# estimate usage frequency using zipf's law and an even mixture of the four corpora
# say we have 1M words from each corpora, composed only of the 5K ranked words...
def float_or_max(_,N=10000):
    if _ == "--":
        return N+1
    else:
        return int(_)  

def get_avg_rank(word):
    # two lines
    ranks = list(map(lambda x: float_or_max(x,N=len(my_labMT.data)),my_labMT.data[word][3:]))
    return np.average(ranks)

sorted_words = sorted([[word,get_avg_rank(word),my_labMT.data[word][1]] for word in my_labMT.data],key=lambda x: x[1])

sorted_words[-10:]

total_words = len(my_labMT.data)

n_iter = 20
remove_amount = int(np.floor(total_words/float(n_iter)))

remove_amount

my_labMT.score({"dull": 1, "happy": 1, "person": 1})

i = 1
for word,r,h in sorted_words[-(i*remove_amount):]:
    del my_labMT.data[word]

len(my_labMT.data)

my_labMT.score({"dull": 1, "happy": 1, "person": 1})

def classifier_perf(conf_mat,v=False):
    """Given the confusion matrix, produce precision, recall, and f1-score.
    Actual going down, predicited going across."""
    
    N = conf_mat.shape[0]
    # could do these computations using matrix math...
    R = np.array([conf_mat[i,i]/conf_mat[i,:].sum() for i in range(N)])
    P = np.array([conf_mat[i,i]/conf_mat[:,i].sum() for i in range(N)])

    F1 = np.array([2*R[i]*P[i]/(R[i]+P[i]) for i in range(N)])
    if v:
        print("R: ",R)
        print("P: ",P)
        print("F1: ",F1)
    return F1.mean()

a = np.array([[.7,.3],[.3,.7]])

a

a.shape

a.mean()

def movie_reviews(sentiment_class,center=5.0,v=False):
    a = np.zeros((2,2))
    for j,flip in enumerate(["pos","neg"]):
        files = ["../data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
                 for x in listdir("../data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
        for file in files:
            f = open(file+".txt","r")
            rawtext = f.read()
            f.close()
            # add to the full dict
            allwordcounts = dictify(listify(rawtext))
            score = sentiment_class.score(allwordcounts)
            if score > center:
                a[j,0] += 1
            else:
                a[j,1] += 1
    if v:
        print(a)
    return classifier_perf(a,v=v)
def movie_reviews_CV(sentiment_class,center=5.0,v=False,cv_count=10):
    '''Cross validated scoring of movie reviews.
    
    Positive are class 0, negative are class 1. (this is j in the loop for movie_reviews)'''
    conf_mats = [np.zeros((2,2)) for _ in range(cv_count)]
    flip = "pos"
    pos_files = ["../data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
             for x in listdir("../data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    flip = "neg"
    neg_files = ["../data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
                 for x in listdir("../data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    # number of samples in each fold
    n_samples = int(np.ceil(1000/cv_count))
    for i in range(cv_count):
        for j,files in enumerate([pos_files,neg_files]):
            for file in files[(i*n_samples):((i+1)*n_samples)]:
                f = open(file+".txt","r")
                rawtext = f.read()
                f.close()
                # add to the full dict
                allwordcounts = dictify(listify(rawtext))
                score = sentiment_class.score(allwordcounts)
                if score > center:
                    conf_mats[i][j,0] += 1
                else:
                    conf_mats[i][j,1] += 1
    totals = np.zeros((2,2))
    for i in range(cv_count):
        totals += conf_mats[i]
    f1_scores = [classifier_perf(totals-conf_mats[i],v=v) for i in range(cv_count)]
    if v:
        print("All F1: ",f1_scores)
    return np.mean(f1_scores),np.std(f1_scores)
def movie_reviews_CV_calibrated(sentiment_class,center=5.0,v=False,train=.1,iter=100):
    '''Cross validated scoring of movie reviews.
    
    Uses the held out 10% to assess a center. (train=hold out for calibration)
    Run this in more "bootstrapped" fashion, for 100 iterations or so. (iter)
    
    Positive are class 0, negative are class 1. (this is j in the loop for movie_reviews)'''
    conf_mats = [np.zeros((2,2)) for _ in range(iter)]
    flip = "pos"
    pos_files = ["../data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
             for x in listdir("../data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    flip = "neg"
    neg_files = ["../data/moviereviews/txt_sentoken/{0}/{1}".format(flip,x.replace(".txt",""))
                 for x in listdir("../data/moviereviews/txt_sentoken/{0}/".format(flip)) if ".txt" in x]
    def score_file(sentiment_class,file):
        f = open(file+".txt","r")
        rawtext = f.read()
        f.close()
        # add to the full dict
        allwordcounts = dictify(listify(rawtext))
        return sentiment_class.score(allwordcounts)
    pos_scores = [score_file(sentiment_class,x) for x in pos_files]
    neg_scores = [score_file(sentiment_class,x) for x in neg_files]
    # number of samples for calibration
    n_samples = int(np.ceil(1000*train))
    for i in range(iter):
        a = np.random.choice(1000,size=n_samples,replace=False)
        b = np.random.choice(1000,size=n_samples,replace=False)
        center = np.mean([pos_scores[x] for x in np.arange(1000) if x not in a]+[neg_scores[x] for x in np.arange(1000) if x not in b])
        for pos_score in [pos_scores[x] for x in a]:
            if pos_score > center:
                conf_mats[i][0,0] += 1
            else:
                conf_mats[i][0,1] += 1
        for neg_score in [neg_scores[x] for x in b]:
            if neg_score > center:
                conf_mats[i][1,0] += 1
            else:
                conf_mats[i][1,1] += 1
    f1_scores = [classifier_perf(conf_mats[i],v=v) for i in range(iter)]
    if v:
        print("All F1: ",f1_scores)
    return np.mean(f1_scores),np.std(f1_scores)

perf_delH = []
for stopVal in tqdm(np.arange(0,3.75,.25)):
    # print(stopVal)
    my_labMT = LabMT(stopVal=stopVal)
    center = my_labMT.scorelist.mean()
    perf_delH.append(movie_reviews(my_labMT,center=center))

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.set_xlabel("Stop value $\Delta h$")
ax.set_ylabel("F1 Mean Score")

perf_delH = []
perf_delH_err = []
for stopVal in tqdm(np.arange(0,3.75,.25)):
    # print(stopVal)
    my_labMT = LabMT(stopVal=stopVal)
    center = my_labMT.scorelist.mean()
    f1,std = movie_reviews_CV_calibrated(my_labMT,center=center)
    perf_delH.append(f1)
    perf_delH_err.append(std)
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.errorbar(np.arange(0,3.75,.25),perf_delH,yerr=perf_delH_err,linewidth=2,color=".1")
ax.set_xlabel("Stop value $\Delta h$")
ax.set_ylabel("F1 Mean Score")

my_labMT = LabMT(stopVal=0.0)

allcountsListSorted,allwordsListSorted = loadMovieReviews()

allcountsListSorted[:10]

allwordsListSorted[:10]

maxCount = 15000
total = np.sum(allcountsListSorted[:maxCount])
def coverageMaker(wordList,sentimentDict):
    a = np.array([sentimentDict.matcherBool(word) for word in wordList[:maxCount]])
    b = np.cumsum(a)/(np.array(range(len(a)))+1)
    return a,b
def totalCoverage(indices):
    return indices*allcountsListSorted[:maxCount]
def covS(indices):
    return np.sum(totalCoverage(indices))/total
labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,my_labMT)
perc_total_cov = covS(labMTcoverage)
perc_words_cov = labMTcovP[-1]

get_avg_rank("the")

get_avg_rank("monster")

labMT_random = LabMT()
sorted_words = sorted([word for word in labMT_random.data],key=lambda x: get_avg_rank(x))
labMT_least = LabMT()
labMT_most = LabMT()

coverage_total = np.zeros((n_iter+1,3))
coverage_words = np.zeros((n_iter+1,3))
f1_scores_LC = np.zeros((n_iter+1,3))
f1_scores_LC_CV = np.zeros((n_iter+1,3))
f1_scores_LC_CV_err = np.zeros((n_iter+1,3))
f1_scores_LC_BS = np.zeros((n_iter+1,3))
f1_scores_LC_BS_err = np.zeros((n_iter+1,3))

sorted_words[:10]

sorted_words[-10:]

f1_scores_LC[0,:] = movie_reviews(my_labMT,center=my_labMT.scorelist.mean())
CV_score,CV_err = movie_reviews_CV(my_labMT,center=my_labMT.scorelist.mean(),v=True)
print(CV_score,CV_err)
f1_scores_LC_CV[0,:] = CV_score
f1_scores_LC_CV_err[0,:] = CV_err
CV_score,CV_err = movie_reviews_CV_calibrated(my_labMT,v=True)
print(CV_score,CV_err)
f1_scores_LC_BS[0,:] = CV_score
f1_scores_LC_BS_err[0,:] = CV_err
labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,my_labMT)
coverage_total[0,:] = covS(labMTcoverage)
n_random_trials = 10

for i in tqdm(range(n_iter)):
    # random
    random_results = np.zeros((7,n_random_trials))
    for j in range(n_random_trials):
        labMT_random = LabMT()
        remove_words = np.random.choice([word for word in labMT_random.data.keys()],size=remove_amount*(i+1),replace=False)
        for word in remove_words:
            del labMT_random.data[word]
        # least freq
        random_results[0,j] = movie_reviews(labMT_random,center=labMT_random.scorelist.mean())
        labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,labMT_random)
        random_results[1,j] = covS(labMTcoverage)
        random_results[2,j] = labMTcovP[-1]
        CV_score,CV_err = movie_reviews_CV(labMT_random,center=my_labMT.scorelist.mean())
        random_results[3,j] = CV_score
        random_results[4,j] = CV_err
        CV_score,CV_err = movie_reviews_CV_calibrated(labMT_random,center=my_labMT.scorelist.mean())
        random_results[5,j] = CV_score
        random_results[6,j] = CV_err
    remove_words = sorted_words[-(i+1)*remove_amount:]
    # print(remove_words[-10:])
    for word in remove_words:
        if word in labMT_least.data:
            del labMT_least.data[word]
    # most freq
    remove_words = sorted_words[0:(i+1)*remove_amount]
    # print(remove_words[:10])
    for word in remove_words:
        if word in labMT_most.data:
            del labMT_most.data[word]
    f1_scores_LC[i+1,0] = random_results[0,:].mean()
    f1_scores_LC_CV[i+1,0] = random_results[3,:].mean()
    f1_scores_LC_CV_err[i+1,0] = random_results[4,:].mean()
    f1_scores_LC_BS[i+1,0] = random_results[5,:].mean()
    f1_scores_LC_BS_err[i+1,0] = random_results[6,:].mean()
    f1_scores_LC[i+1,1] = movie_reviews(labMT_least,center=labMT_least.scorelist.mean())
    f1_scores_LC[i+1,2] = movie_reviews(labMT_most,center=labMT_most.scorelist.mean())
    CV_score,CV_err = movie_reviews_CV(labMT_least,center=my_labMT.scorelist.mean())
    f1_scores_LC_CV[i+1,1] = CV_score
    f1_scores_LC_CV_err[i+1,1] = CV_err
    CV_score,CV_err = movie_reviews_CV(labMT_most,center=my_labMT.scorelist.mean())
    f1_scores_LC_CV[i+1,2] = CV_score
    f1_scores_LC_CV_err[i+1,2] = CV_err
    CV_score,CV_err = movie_reviews_CV_calibrated(labMT_least,center=my_labMT.scorelist.mean())
    f1_scores_LC_BS[i+1,1] = CV_score
    f1_scores_LC_BS_err[i+1,1] = CV_err
    CV_score,CV_err = movie_reviews_CV_calibrated(labMT_most,center=my_labMT.scorelist.mean())
    f1_scores_LC_BS[i+1,2] = CV_score
    f1_scores_LC_BS_err[i+1,2] = CV_err
    coverage_total[i+1,0] = random_results[1,:].mean()
    coverage_words[i+1,0] = random_results[2,:].mean()
    labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,labMT_least)
    coverage_total[i+1,1] = covS(labMTcoverage)
    coverage_words[i+1,1] = labMTcovP[-1]
    labMTcoverage,labMTcovP = coverageMaker(allwordsListSorted,labMT_most)
    coverage_total[i+1,2] = covS(labMTcoverage)
    coverage_words[i+1,2] = labMTcovP[-1]

f1_scores_LC_CV

remove_amount*np.arange(n_iter+1)/10222

len(labMT_least.data)

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Percentage words removed")
ax.set_ylabel("F1 Mean Score")
plt.legend(["Random","Least freq","Most freq"],loc="best")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_CV[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_CV[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_CV[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
plt.legend(["Random","Least freq","Most freq"],loc="best")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-CV.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-CV.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-001.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-000.pdf")
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
ax.set_ylim([0,.66])
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-001.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-001.pdf")
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
ax.set_ylim([0,1])
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-002.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-002.pdf")
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
ax.set_ylim([.5,1])
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-003.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-003.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],yerr=f1_scores_LC_BS_err[:,0],linewidth=2,color=".05")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222+.01,f1_scores_LC_BS[:,1],yerr=f1_scores_LC_BS_err[:,1],linewidth=2,color=".45")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222+.02,f1_scores_LC_BS[:,2],yerr=f1_scores_LC_BS_err[:,2],linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
ax.set_xlim([0,1.03])
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-000.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-EB-000.pdf")
fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222,f1_scores_LC_BS[:,0],yerr=f1_scores_LC_BS_err[:,0],linewidth=2,color=".05")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222+.01,f1_scores_LC_BS[:,1],yerr=f1_scores_LC_BS_err[:,1],linewidth=2,color=".45")
ax.errorbar(remove_amount*np.arange(n_iter+1)/10222+.02,f1_scores_LC_BS[:,2],yerr=f1_scores_LC_BS_err[:,2],linewidth=2,color=".75")
ax.set_xlabel("Fraction words removed")
ax.set_ylabel("F1 Mean Score")
ax.set_ylim([0,1])
ax.set_xlim([0,1.03])
plt.legend(["Random","Least freq","Most freq"],loc="best")
# mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-001.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-result-BS-EB-001.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
# ax.plot(np.arange(0,3.75,.25),perf_delH,"-",linewidth=2,color=".1")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,coverage_total[:,0],"-",linewidth=2,color=".05")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,coverage_total[:,1],"-",linewidth=2,color=".45")
ax.plot(remove_amount*np.arange(n_iter+1)/10222,coverage_total[:,2],"-",linewidth=2,color=".75")
ax.set_xlabel("Percentage words removed")
ax.set_ylabel("Total coverage")
ax.set_ylim([0,1])
plt.legend(["Random","Least freq","Most freq"],loc="best")
mysavefig("../figures/labMT-test/labMT-coverage-removal-coverage.png")
mysavefig("../figures/labMT-test/labMT-coverage-removal-coverage.pdf")

plt.plot(coverage_words[1:,:])
plt.legend(["Random","Least freq","Most freq"])

labMT_binary = LabMT(stopVal=1.0)
labMT_binary_center = labMT_binary.scorelist.mean()
labMT_binary_center = 5.0
for word,score in zip(labMT_binary.wordlist,labMT_binary.scorelist):
    labMT_binary.data[word] = list(labMT_binary.data[word])
    if score >= labMT_binary_center:
        labMT_binary.data[word][1] = 1
    else:
        labMT_binary.data[word][1] = -1

labMT_binary.data["laughter"]

new_scores = np.array([labMT_binary.data[word][1] for word in labMT_binary.data])

new_scores.mean()

movie_reviews(labMT_binary,center=0.0,v=True)

len(labMT_binary.data)

movie_reviews(labMT_binary,center=new_scores.mean(),v=True)

binarizations = [0.0,.25,.5,.75,1.0]
labMT_binary_list = [LabMT(stopVal=0.0) for i in range(len(binarizations))]
# distribution center
labMT_binary_center = labMT_binary_list[0].scorelist.mean()
# real center
labMT_binary_center = 5.0
for i in range(len(binarizations)):
    for word,score in zip(labMT_binary_list[i].wordlist,labMT_binary_list[i].scorelist):
        labMT_binary_list[i].data[word] = list(labMT_binary_list[i].data[word])
        if score >= labMT_binary_center:
            labMT_binary_list[i].data[word][1] = ((labMT_binary_list[i].data[word][1]-labMT_binary_center)-1)*(1-binarizations[i])+1
        elif score < labMT_binary_center:
            labMT_binary_list[i].data[word][1] = ((labMT_binary_list[i].data[word][1]-labMT_binary_center)+1)*(1-binarizations[i])-1

# all the same length, but now the max scores are being toned down
[max([x.data[y][1] for y in x.data]) for x in labMT_binary_list]

[len(x.data) for x in labMT_binary_list]

for i in range(len(binarizations)):
    for j,word in enumerate(labMT_binary_list[0].wordlist):
        labMT_binary_list[i].scorelist[j] = labMT_binary_list[i].data[word][1]

[max(x.scorelist) for x in labMT_binary_list]

f1_scores = np.zeros(len(binarizations))
for i in tqdm(range(len(binarizations))):
    f1_scores[i] = movie_reviews(labMT_binary_list[i],center=0.0,v=True)
f1_scores

f1_scores_BI = np.zeros(len(binarizations))
f1_scores_BI_CV = np.zeros(len(binarizations))
f1_scores_BI_CV_err = np.zeros(len(binarizations))
for i in tqdm(range(len(binarizations))):
    f1_scores_BI[i] = movie_reviews(labMT_binary_list[i],center=labMT_binary_list[i].scorelist.mean(),v=False)
    CV_score,CV_err = movie_reviews_CV_calibrated(labMT_binary_list[i])
    f1_scores_BI_CV[i] = CV_score
    f1_scores_BI_CV_err[i] = CV_err

f1_scores_BI

f1_scores_BI_CV

f1_scores_BI_CV_err

plt.plot(binarizations,f1_scores_BI)
plt.ylim([.5,.7])
plt.xlabel("Percentage binarized")
plt.ylabel("F1 Score")
# mysavefig("labMT_binarization_f1.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.plot(binarizations,f1_scores_BI,"-",linewidth=2,color=".15")
ax.set_ylim([.5,.7])
ax.set_xlabel("Percentage binarized")
ax.set_ylabel("F1 Score")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.bar(np.array(binarizations)-.1,f1_scores_BI,width=.2,color=".85")
ax.set_ylim([.5,.75])
ax.set_xlim([-.15,1.15])
ax.set_xlabel("Percentage binarized")
ax.set_ylabel("F1 Score")
mysavefig("../figures/labMT-test/labMT-binary-performace-bars.png")
mysavefig("../figures/labMT-test/labMT-binary-performace-bars.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.bar(np.array(binarizations)-.1,f1_scores_BI_CV,yerr=f1_scores_BI_CV_err,width=.2,color=".85",ecolor="k")
ax.set_ylim([.0,.7])
ax.set_xlim([-.15,1.15])
ax.set_xlabel("Percentage binarized")
ax.set_ylabel("F1 Score")
mysavefig("../figures/labMT-test/labMT-binary-performace-bars-CV.pdf")

fig = plt.figure()
ax = fig.add_axes([.2,.2,.7,.7])
ax.bar(np.array(binarizations)-.1,f1_scores_BI,width=.2,color=".85")
ax.set_ylim([0,.65])
ax.set_xlim([-.15,1.15])
ax.set_xlabel("Percentage binarized")
ax.set_ylabel("F1 Score")
mysavefig("../figures/labMT-test/labMT-binary-performace-bars-ylim.png")
mysavefig("../figures/labMT-test/labMT-binary-performace-bars-ylim.pdf")

# stopVals = np.arange(0,3.5,.25)
# centers = np.arange(4.5,6.5,.05)
# perf = np.zeros((len(stopVals),len(centers)))
# for i,stopVal in tqdm(enumerate(stopVals)):
#     # print(stopVal)
#     my_labMT = LabMT(stopVal=stopVal)
#     # print(my_labMT.scorelist.mean())
#     for j,center in tqdm(enumerate(centers)):
#         perf[i,j] = movie_reviews(my_labMT,center=center,v=False)

# fig = plt.figure(figsize=(10,8))
# ax = fig.add_axes([.2,.2,.7,.7])
# # heatmap = ax.pcolor(perf, cmap=plt.cm.Blues)

# # put the major ticks at the middle of each cell
# # ax.set_xticks(np.arange(perf.shape[0])+0.5, minor=False)
# # ax.set_yticks(np.arange(perf.shape[1])+0.5, minor=False)

# masked_array = np.ma.array(perf.transpose(), mask=np.isnan(perf.transpose()))
# cmap = plt.cm.Blues
# cmap.set_bad('white',1.)
# heatmap = ax.pcolor(masked_array, cmap=cmap)
# ax.set_xticks(np.arange(perf.shape[0])+0.5, minor=False)
# ax.set_xticklabels(stopVals)
# ax.set_yticks(np.arange(perf.shape[1],step=3)+0.5, minor=False)
# ax.set_yticklabels(centers[0::3])
# bar = fig.colorbar(heatmap, extend='both')
# ax.set_xlabel("Delta h")
# ax.set_ylabel("Decision boundary")
# plt.savefig("../figures/labMT-test/optimal-boundary-vs-delH-heatmap.pdf")

# plt.imshow(perf)

# fig = plt.figure(figsize=(7,3.5))
fig = plt.figure()
with plt.style.context("paper-twocol"):
    ax = fig.add_axes([.06,.06+.35,.3,.93])
    # ax.bar(np.array(binarizations)-.1,f1_scores_BI,width=.2,color=".85")
    ax.bar([-.1,.15],[f1_scores_BI[0],f1_scores_BI[-1]],width=.2,color=".85")
    ax.set_ylim([0,1])
    ax.set_xlim([-.15,.4])
    ax.set_xticks([0,.25])
    ax.set_xticklabels(["Continuous","Binary"])
    ax.set_xlabel("labMT")
    ax.set_ylabel("F1 Score")
    mysavefig("../figures/labMT-test/labMT-binary-performace-bars-simple.png")
    mysavefig("../figures/labMT-test/labMT-binary-performace-bars-simple.pdf")

from IPython.core.display import HTML,Javascript

# # this is the header var in the function we're about to call
# HTML(r"""<link href="static/css/hedotools.shift.css" rel="stylesheet">
# <script src="static/js/d3.v4.js" charset="utf-8"></script>
# <script src="static/js/jquery-1.11.0.min.js" charset="utf-8"></script>
# <script src="static/js/urllib.js" charset="utf-8"></script>
# <script src="static/js/hedotools.init.js" charset="utf-8"></script>
# <script src="static/js/hedotools.shifter.js" charset="utf-8"></script>""")

get_ipython().run_cell_magic('HTML', '', '<link href="static/css/hedotools.shift.css" rel="stylesheet">\n<script src="static/js/d3.v4.js" charset="utf-8"></script>\n<script src="static/js/jquery-1.11.0.min.js" charset="utf-8"></script>\n<script src="static/js/urllib.js" charset="utf-8"></script>\n<script src="static/js/hedotools.init.js" charset="utf-8"></script>\n<script src="static/js/hedotools.shifter.js" charset="utf-8"></script>')

# shiftHtmlJupyter(my_LabMT.scorelist,my_LabMT.wordlist,control_wordvec_s,diag_wordvec_s,"depression/all.html",
#                  corpus="LabMT",advanced=False,customTitle=True,
#                  title="Depression versus healthy tweets",ref_name="healthy tweets",comp_name="depression",
#                  ref_name_happs="Healthy tweet",comp_name_happs="Depression tweet",isare=" are ")
# HTML(open("depression/all.html").read())

get_ipython().run_cell_magic('javascript', '', "// require.config({\n//   paths: {\n//       d3: 'static/js/d3.v4.js'\n//   }\n// });")

