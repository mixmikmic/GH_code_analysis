import sys
sys.path.append("/Users/andyreagan/tools/python/labMTsimple/")
from labMTsimple.speedy import *
from labMTsimple.storyLab import *

import re
import codecs
from os import listdir,mkdir
from os.path import isfile,isdir
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib import rc,rcParams
rc("xtick", labelsize=8)
rc("ytick", labelsize=8)
rc("font",**{"family":"serif","serif":["cmr10"]})
# rc("text", usetex=True)
figwidth_onecol = 8.5
figwidth_twocol = figwidth_onecol/2

import numpy as np
from json import loads
import csv
from datetime import datetime,timedelta
import pickle

error_logging = True
sys.path.append("/Users/andyreagan/tools/python/kitchentable")
from dogtoys import *

def scatter_sections_all(all_happs_arrays,sections,allDicts):
    """Make a scatterplot of NYT sections.

    Needs the happs arrays from each dictionary, and the names of the sections.
    all_happs_arrays = [[],[],...] with each [] the list of happs for each section for that dictionary.
    sections = titles of the sections."""
    
    ranges = np.array([8,8,8,2,2,2])
    centers = np.array([5,5,5,0,0,0])

    import scipy.odr.odrpack as odrpack

    def f(B, x):
        return B[0]*x + B[1]

    linear = odrpack.Model(f)

    # avg_happs_unweighted = my_happs.mean()
    fig = plt.figure(figsize=(12,12))
    # ax = fig.add_axes([.2,.2,.7,.7])
    # going across the bottom
    for i,happs_1 in enumerate(all_happs_arrays[:-1]):
        # going down from the top
        for j,happs_2 in enumerate(all_happs_arrays[i+1:]):
            print(i,j)
            ax = plt.subplot(len(all_happs_arrays)-1,len(all_happs_arrays)-1,i+j*5+1+i*5)
            if i+j*5+1+i*5 > 20:
                plt.xlabel(allDicts[i].title)
            else:
                ax.set_xticklabels([])
            if np.mod(i+j*5+i*5,5) == 0:
                ax.set_ylabel(allDicts[j+i+1].title)
            else:
                ax.set_yticklabels([])

            happs_1_norm = (happs_1-centers[i])/(ranges[i]/2.0)
            happs_2_norm = (happs_2-centers[j+i+1])/(ranges[j+i+1]/2.0)
            plt.scatter(happs_1_norm,happs_2_norm)
            for k,section in enumerate(sections):
                plt.annotate(section,(happs_1_norm[k]+.005,happs_2_norm[k]+.005),fontsize=10)

            mydata = odrpack.RealData(happs_1_norm, happs_2_norm) # sx=set1scoresStd, sy=set2scoresStd)

            myodr = odrpack.ODR(mydata, linear, beta0=[1., 0.])
            myoutput = myodr.run()
            myoutput.pprint()
            # print(myoutput.beta)
            # print(myoutput.sd_beta)

            # ax.scatter(set1scores,set2scores,alpha=0.9,marker="o",c="#F0F0FA",s=12,linewidth=0.0,edgecolor="k")


            x = np.linspace(min(happs_1_norm),max(happs_1_norm),num=10)
            ax.plot(x,myoutput.beta[0]*x+myoutput.beta[1],"r",linewidth=0.75)

            ax.legend(["RMA\n$\\beta$ = {0:.2f}\n$\\alpha$ = {1:.2f}".format(myoutput.beta[0],myoutput.beta[1])],loc="best",fontsize=10,frameon=False,)

    plt.tight_layout(pad=0.5, w_pad=0.0, h_pad=0.0)
    for i,happs_1 in enumerate(all_happs_arrays[:-1]):
        # going down from the top
        for j,happs_2 in enumerate(all_happs_arrays[i+1:]):
            print(i,j)
            ax = plt.subplot(len(all_happs_arrays)-1,len(all_happs_arrays)-1,i+j*5+1+i*5)
            if i+j*5+1+i*5 > 20:
                plt.xlabel(allDicts[i].title,fontsize=15)
            else:
                ax.set_xticklabels([])
            if np.mod(i+j*5+i*5,5) == 0:
                ax.set_ylabel(allDicts[j+i+1].title,fontsize=15)
            else:
                ax.set_yticklabels([])
            # plt.scatter(happs_1,happs_2)
            # for k,section in enumerate(sections):
            #    plt.annotate(section,(happs_1[k],happs_2[k]))
    mysavefig("NYT-scatter-all.png",folder="../figures/nyt")
    mysavefig("NYT-scatter-all.pdf",folder="../figures/nyt")

def nyt_sorted_barchart(sections,my_senti_dict,my_happs):

    indexer = sorted(range(len(my_happs)),key=lambda x: my_happs[x],reverse=True)

    sections_sorted = [sections[i] for i in indexer]
    my_happs_sorted = [my_happs[i] for i in indexer]
    together_sorted = [(sections[i],my_happs[i]) for i in indexer]

    avg_happs_unweighted = my_happs.mean()

    fig = plt.figure(figsize=(8,12))
    ax = fig.add_axes([.2,.2,.7,.7])
    # ax.bar(np.arange(len(sections)),np.array(my_happs_sorted),orientation="vertical")
    # ax.bar(0,.8,np.array(my_happs_sorted)-avg_happs,bottom=np.arange(len(sections)),orientation="horizontal")
    bar_height = 0.8
    happs_diff = np.array(my_happs_sorted)-avg_happs_unweighted
    rects1 = ax.bar(0,bar_height,happs_diff,bottom=np.arange(len(sections)),orientation="horizontal")
    ax.set_ylim([bar_height-1,len(sections)])
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xlabel("Happs diff from unweighted average")
    # ax.set_title("Ranking of NYT Sections by {0}".format(my_senti_dict.title))

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            width = rect.get_width()
            y = rect.get_y()

            x = rect.get_x()
            if happs_diff[y] > 0:
                ax.text(-0.01, y+bar_height/2., "{0:.0f}. {1}".format(y+1,sections_sorted[int(np.floor(y))].replace("_"," ").capitalize()),
                        ha="right", va="center")
            else:
                ax.text(.01, y+bar_height/2., "{0:.0f}. {1}".format(y+1,sections_sorted[int(np.floor(y))].replace("_"," ").capitalize()),
                        ha="left", va="center")

    autolabel(rects1)

    # mysavefig("NYT-sorted-{0}.png".format(my_senti_dict.title),date=False)
    mysavefig("NYT-sorted-{0}.pdf".format(my_senti_dict.title),date=True,folder="../figures/nyt")

def nyt_happs(sections,my_senti_dict,stopVal = 1.0,make_word_shifts=True,prefix="",use_cache=True):
    """Do everything related to the NYT sections.

    sections: list of section titles
    my_senti_dict: an individual sentiDict subclass
    """
    happs_array = np.zeros(len(sections))
    total_word_vec = np.zeros(len(my_senti_dict.scorelist))
    for i,section in enumerate(sections):
        print(section)
        # check for the csv
        fname = "../data/nyt/sections/NYT_{0}-{1}.csv".format(section,my_senti_dict.title)
        my_word_vec = np.zeros(len(my_senti_dict.scorelist))
        if isfile(fname) and use_cache:
            print("using cache")
            f = open(fname,"r")
            my_word_vec = np.array(list(map(float,f.read().split("\n"))))
            f.close()
        else:
            print("loading dict directly")
            raw_fname = "../data/nyt/sections/NYT_{0}.dict".format(section)
            my_raw_dict = pickle.load( open( raw_fname , "rb" ) )
            my_word_vec = my_senti_dict.wordVecify(my_raw_dict)
            # may not want to either load the word vec,
            # or save a new one, when doing tests of removing stop words
            # should probably just use the stopper for this.
            if use_cache:
                print("saving cache")
                f = open( fname , "w" )
                f.write( "\n".join( list( map( lambda x: "{0:.0f}".format(x) , my_word_vec ) ) ) )
                f.close()
        print(len(my_word_vec))
        print(len(my_senti_dict.stopper(my_word_vec,stopVal=stopVal)))
        my_word_vec = my_senti_dict.stopper(my_word_vec,stopVal=stopVal)
        total_word_vec += my_word_vec

        happs = np.dot(my_word_vec,my_senti_dict.scorelist)/np.sum(my_word_vec)
        print(happs)
        happs_array[i] = happs
   
    avg_happs = np.dot(total_word_vec,my_senti_dict.scorelist)/np.sum(total_word_vec)

    nyt_sorted_barchart(sections,my_senti_dict,happs_array)

    total_word_vec_stopped = my_senti_dict.stopper(total_word_vec)

    if make_word_shifts:
        for i,section in enumerate(sections):
            print(section)
            fname = "../data/nyt/sections/NYT_{0}-{1}.csv".format(section,my_senti_dict.title)
            f = open(fname,"r")
            my_word_vec = np.array(map(int,f.read().split("\n")))
            f.close()
            my_word_vec_stopped = my_senti_dict.stopper(my_word_vec)
            shiftHtml(my_senti_dict.scorelist, my_senti_dict.wordlist, 
                    total_word_vec_stopped, my_word_vec_stopped, 
                    "NYT-shift-{0}-{1}.html".format(my_senti_dict.title,section),
                    # make_png_too=False,open_pdf=False,
                    customTitle=True,
                    title="{0}{1} Wordshift".format(prefix,my_senti_dict.title),
                    ref_name="NYT as a whole",comp_name=section+" section",
                    ref_name_happs="NYT as a whole",comp_name_happs=(section+" section").capitalize())
            # (scoreList, wordList, refFreq, compFreq, outFile)

    return (happs_array,avg_happs)

def shift_NYT_zipf():
    import scipy.odr.odrpack as odrpack

    def f(B, x):
        return B[0] * (x ** (B[1]))

    power_law = odrpack.Model(f)

    a = pickle.load( open("../data/nyt/sections/NYT_society.dict", "rb") )
    b = pickle.load( open("../data/nyt/all-50k.dict", "rb") )

    words = [word for word in b]
    indexer = sorted(range(len(words)),key=lambda k: b[words[k]], reverse=True)
    sorted_words = [words[i] for i in indexer[:50000]]
    sorted_counts = [b[word] for word in sorted_words]

    society_counts = [a[word] if word in a else 0 for word in sorted_words]

    print(sorted_counts[0])
    mydata = odrpack.RealData(range(1,len(sorted_counts)+1),sorted_counts) # sx=set1scoresStd, sy=set2scoresStd)
    myodr = odrpack.ODR(mydata, power_law, beta0=[sorted_counts[0], -1.])
    myoutput = myodr.run()
    myoutput.pprint()

    # b50k = dict()
    # for word,count in zip(sorted_words,sorted_counts):
    #     b50k[word] = count
    # pickle.dump( b50k , open("data/nyt/all-50k.dict", "wb") )

    # zipf_weights  = [1.0/(i+1.0) for i in range(len(sorted_words))]
    # shiftHtml(zipf_weights,sorted_words,sorted_counts,society_counts,"society-zipf.html")

    # inverse_zipf_weights  = [i+1.0 for i in range(len(sorted_words))]
    # shiftHtml(inverse_zipf_weights,sorted_words,sorted_counts,society_counts,"society-zipf-inv.html")

    # inverse_counts = [1.0/count for count in sorted_counts]
    # shiftHtml(inverse_counts,sorted_words,sorted_counts,society_counts,"society-counts-inv.html")

    inverse_zipf_weights_fitted  = [1.0/(myoutput.beta[0]*( (i+1) ** myoutput.beta[1] )) for i in range(len(sorted_words))]
    print(inverse_zipf_weights_fitted[:10])
    shiftHtml(inverse_zipf_weights_fitted,sorted_words,sorted_counts,society_counts,"society-zipf-inv-fit.html")

def marriage_words():
    a = pickle.load( open("../data/nyt/sections/NYT_society.dict", "rb") )
    matches = []
    for word in a:
        if word[0:3] == "mar":
            matches.append((word,a[word]))
    indexer = sorted(range(len(matches)),key=lambda k: matches[k][1],reverse=True)
    sorted_matches = [matches[i] for i in indexer]
    for i in range(6):
        print(sorted_matches[i][0]+" ")
        print("("+str(sorted_matches[i][1])+"), ")

shift_NYT_zipf()

sections = ["arts","books","classified","cultural","editorial","education","financial","foreign","home","leisure","living","magazine","metropolitan","movies","national","regional","science","society","sports","style","television","travel","week-in-review","weekend",]

stopVal = 0.0
allDicts = (LabMT(stopVal=stopVal),
            ANEW(stopVal=stopVal),
            WK(stopVal=stopVal),
            MPQA(stopVal=stopVal),
            LIWC(stopVal=stopVal),
            Liu(stopVal=stopVal))


all_stopVals = [1.0,1.0,1.0,0.5,0.5,0.5]
all_happs_arrays = [[] for i in range(len(allDicts))]
for k,my_dict in enumerate(allDicts):
# all_stopVals = [0.5,]
# for i,my_dict in enumerate([allDicts[4]]):
    print("#"*80)
    print(my_dict.title)
    print("#"*80)
    my_happs_array,avg_happs = nyt_happs(sections,
                                         my_dict,
                                         stopVal=all_stopVals[k],
                                         make_word_shifts=False,
                                         prefix=letters[k]+": ",
                                         use_cache=True)
    all_happs_arrays[k] = my_happs_array


scatter_sections_all(all_happs_arrays,sections,allDicts)

# fix Liu
for word in ["vice","miss"]:
    if word in allDicts[5].data:
        print("removing "+word)
        allDicts[5].scorelist[allDicts[5].data[word][0]] = 0.0
    else:
        print(word+" not in Liu")

for word in ["mar*","retire*","vice","bar*","miss*"]:
    if word in allDicts[3].data:
        print("removing "+word)
        allDicts[3].scorelist[allDicts[3].data[word][0]] = 0.0
    else:
        print(word+" not in MPQA")

for k,my_dict in enumerate(allDicts):
# all_stopVals = [0.5,]
# for i,my_dict in enumerate([allDicts[4]]):
    print("#"*80)
    print(my_dict.title)
    print("#"*80)
    my_happs_array,avg_happs = nyt_happs(sections,
                                         my_dict,
                                         stopVal=all_stopVals[k],
                                         make_word_shifts=False,
                                         prefix=letters[k]+": ",
                                         use_cache=False)
    all_happs_arrays[k] = my_happs_array



