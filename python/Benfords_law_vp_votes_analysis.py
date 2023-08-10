import pandas as pd
import numpy as np
import math
import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.ticker as tkr

from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as font_manager
from matplotlib import rcParams

from scipy.stats import chisquare

import scipy.stats as ss
get_ipython().magic('matplotlib inline')

#read data from csv
data = pd.read_csv('VP_election_tallies.csv', parse_dates=[["date", "time"]])
data.sort_values(['region', 'candidate', 'date_time'], inplace=True)

#computed transmitted data
data = data[['date_time', 'candidate', 'region', 'count', 'increment']]
data['count_shift'] = data['count'].diff(periods=1)
data['ref'] = (data.candidate == data.candidate.shift(1)) & (data.region == data.region.shift(1)) 
data['transmitted_votes'] = data.count_shift*data.ref + data['count']*(~data.ref) 
print data[['candidate', 'region', 'count', 'count_shift', 'ref', 'transmitted_votes']][97:110]
data = data[(data.transmitted_votes>0)]

#define cut (2016/5/9 3am)
cut = datetime.datetime(2016, 5, 10, 3, 0)

#clean data and separate at the defined cut
data_after_3am = data[(data.date_time>=cut)&(data.transmitted_votes>0)]
data_before_3am = data[(data.date_time<cut)&(data.transmitted_votes>0)]

#get votes
votes_before_3am = data_before_3am['transmitted_votes'].tolist()
votes_after_3am = data_after_3am['transmitted_votes'].tolist()
votes_before_3am = data_before_3am['transmitted_votes'].tolist()
votes_after_3am = data_after_3am['transmitted_votes'].tolist()
votes_leni_before_3am = data_before_3am[data_before_3am.candidate=='Leni']['transmitted_votes'].tolist()
votes_leni_after_3am = data_after_3am[data_after_3am.candidate=='Leni']['transmitted_votes'].tolist()
votes_bbm_before_3am = data_before_3am[data_before_3am.candidate=='BBM']['transmitted_votes'].tolist()
votes_bbm_after_3am = data_after_3am[data_after_3am.candidate=='BBM']['transmitted_votes'].tolist()

#define digit retriever
def get_digits(votes):
    """Retrieves first and last digit of a number"""
    first_digits = []
    last_digits = []
    for i in votes:
        first_digits.append(int(str(int(i))[0]))
        last_digits.append(int(str(int(i))[-1]))
    return first_digits, last_digits

#store first and last digits of the votes
first_digits_before_3am, last_digits_before_3am = get_digits(votes_before_3am)
first_digits_after_3am, last_digits_after_3am = get_digits(votes_after_3am)
first_digits_leni_before_3am, last_digits_leni_before_3am = get_digits(votes_leni_before_3am)
first_digits_leni_after_3am, last_digits_leni_after_3am = get_digits(votes_leni_after_3am)
first_digits_bbm_before_3am, last_digits_bbm_before_3am = get_digits(votes_bbm_before_3am)
first_digits_bbm_after_3am, last_digits_bbm_after_3am = get_digits(votes_bbm_after_3am)

#input expected normalized frequencies
n_expected_last = [1./9 for i in range(10)]
n_expected_first = [math.log((1.0 + 1.0/i), 10) for i in range(1,10)]

#define plotter
minorLocatorx   = AutoMinorLocator(10)
minorLocatory   = AutoMinorLocator(4)
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
matplotlib.rcParams['axes.linewidth'] = 2.
plt.rcParams['axes.linewidth'] = 4
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman') 

def plot_first_distrution(first_digits, title, n_expected, label=None, show=True):
    xval = [0.5 + i for i in range(1,10)]
    n, y, z = plt.hist(first_digits, bins=range(1,11), normed=1, color=['pink'], lw=1.5, 
                       label='Observed',histtype='stepfilled')
    plt.plot(xval, n_expected, 'g-', lw=3.5, label='Benford')
    plt.xlabel('First Digit', fontsize=20)
    plt.ylabel('Normalized Frequency', fontsize=20)
    plt.xticks(xval,range(1,10))
    plt.ylim(0, np.max(n_expected)+0.15)
    count = np.array(n)*len(first_digits)
    if label:
        plt.text(4.5, 0.26, label, fontsize=16)
    n_groups = len(n)
    label = tuple([str(i) for i in range(1,10)])
    err = tuple([0]*9)
    plt.legend(frameon=0, prop={'size':20})
    plt.tight_layout()
    print chisquare(n, f_exp=n_expected)
    if show:
        plt.show()
    else:
        plt.savefig(title,dpi=400)
    plt.clf()
    
    
def plot_last_distrution(last_digits, title, n_expected, label=None, show=True):
    xval = [i for i in range(1,11)]
    n, y, z = plt.hist(last_digits, bins=range(1,11), normed=1, color=['pink'], lw=1.5, 
                       label='Observed', histtype='stepfilled')
    plt.plot(xval, n_expected, 'g-', lw=3.5, label='Uniform')
    plt.xlabel('Last Digit', fontsize=20)
    plt.ylabel('Normalized Frequency', fontsize=20)
    plt.ylim(0, np.max(n_expected)+0.15)
    plt.xticks([0.5 + i for i in range(1,10)],range(1,10))
    if label:
        plt.text(4, 0.02, label, fontsize=16)
    n_groups = len(n)
    label = tuple([str(i) for i in range(1,10)])
    err = tuple([0]*9)
    plt.legend(frameon=0, prop={'size':20})
    plt.tight_layout()    
    print chisquare(n, f_exp=n_expected[:-1])
    if show:
        plt.show()
    else:
        plt.savefig(title,dpi=400)
    plt.clf()

print "first and last digit distribution of the votes transmitted"
plot_first_distrution(first_digits_before_3am + first_digits_after_3am, 'FirstDigitsDist_FullSample', n_expected_first)
plot_last_distrution(last_digits_before_3am + last_digits_after_3am, 'LastDigitsDist_FullSample', n_expected_last)

print "first and last digit distribution of the votes transmitted before 3am"
plot_first_distrution(first_digits_before_3am, 'FirstDigitsDist_pre3AM', n_expected_first)
plot_last_distrution(last_digits_before_3am, 'LastDigitsDist_pre3AM', n_expected_last)

#first and last digit distribution of the votes transmitted after 3am
plot_first_distrution(first_digits_after_3am, 'FirstDigitsDist_post3AM', n_expected_first)
plot_last_distrution(last_digits_after_3am, 'LastDigitsDist_post3AM', n_expected_last)

#first and last digit distribution of Leni's votes before 3AM
plot_first_distrution(first_digits_leni_before_3am, 'FirstDigitsDist_pre3AM_leni', n_expected_first)
plot_last_distrution(last_digits_leni_before_3am, 'LastDigitsDist_pre3AM_leni', n_expected_last)

#first and last digit distribution of Leni's votes after 3AM
plot_first_distrution(first_digits_leni_after_3am, 'FirstDigitsDist_post3AM_leni', n_expected_first)
plot_last_distrution(last_digits_leni_after_3am, 'LastDigitsDist_post3AM_leni', n_expected_last)

print "first and last digit distribution of BBM's votes before 3AM"
plot_first_distrution(first_digits_bbm_before_3am, 'FirstDigitsDist_pre3AM_bbm', n_expected_first)
plot_last_distrution(last_digits_bbm_before_3am, 'LastDigitsDist_pre3AM_bbm', n_expected_last)

print "first and last digit distribution of BBM's votes after 3AM"
plot_first_distrution(first_digits_bbm_after_3am, 'FirstDigitsDist_post3AM_bbm', n_expected_first)
plot_last_distrution(last_digits_bbm_after_3am, 'LastDigitsDist_post3AM_bbm', n_expected_last)

#generate time partitions
cut_list = [datetime.datetime(2016, 5, 9, 20, 0)]
for i in range(11):
    cut_list.append(cut_list[-1] + datetime.timedelta(hours=2))

#data in time partitions    
partitioned_data = []
for i in range(1, len(cut_list)):
    partitioned_data.append(data[(data.date_time>=cut_list[i-1])&(data.date_time<cut_list[i])])
    
#label partitions
pre_label = np.array([_.strftime("%B %d %I%p") for _ in cut_list])
label = ['{0} to {1}'.format(pre_label[i-1], pre_label[i]) for i in range(1,len(pre_label))]

#generate gif
num = 0
for i in partitioned_data:
    print num
    votes = i[i.increment>0]['increment'].tolist()
    first_digits, last_digits = get_digits(votes)
    plot_first_distrution(first_digits, 'first_{0}'.format(num), n_expected_first, label=label[num])
    num = num + 1



