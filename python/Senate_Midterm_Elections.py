# special IPython command to prepare the notebook for matplotlib
get_ipython().magic('matplotlib inline')

import requests 
from StringIO import StringIO
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operations
import collections

# special matplotlib argument for improved plots
from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

### Your code here ###
url_str = "http://elections.huffingtonpost.com/pollster/api/charts/?topic=2014-senate"
info = requests.get(url_str).json()
type(info)

states = [str(election['state']) for election in info]
titles = [str(election['title']) for election in info]

countStates = collections.Counter(states)
print countStates

twoPollIndex = [elem[0] for elem in countStates.most_common(4)]
twoPollIndex

for state in twoPollIndex:
    ind = [i for i, x in enumerate(states) if x == state]
    for elem in ind:
        print "title: %s (index: %g)" % (titles[elem], elem)

removeset = set([29])
infoClean = [v for i, v in enumerate(info) if i not in removeset]

election_urls = [election['url'] + '.csv' for election in infoClean]

def build_frame(url):
    """
    Returns a pandas DataFrame object containing
    the data returned from the given url
    """
    source = requests.get(url).text
    
    # Use StringIO because pd.DataFrame.from_csv requires .read() method
    s = StringIO(source)
    
    return pd.DataFrame.from_csv(s, index_col=None).convert_objects(
            convert_dates="coerce", convert_numeric=True)

# Makes a dictionary of pandas DataFrames keyed on election string.
polls = dict((election['slug'], build_frame(election['url']+'.csv')) for election in infoClean)

polls.keys()[0:5]

### Your code here ###
rows = []
for i,dt in enumerate(infoClean):
    x = dt['estimates'][0:2]
    if x:
        if not x[0]['last_name']:
            tmp = dt['url'].split('-')+['vs']
            j = tmp.index('vs')
            if j!=len(tmp)-1:
                R = tmp[j-1].capitalize()
                D = tmp[j+1].capitalize()
                incumbent = np.nan
                #if no data means race is decided
        else:
            tmp = [x[0]['party'],x[1]['party']]
            R = x[tmp.index('Rep')]['last_name']
            idx = [k for k in range(len(tmp)) if tmp[k]!='Rep'][0]
            D = x[idx]['last_name']
            tmp2 = [x[0]['incumbent'],x[1]['incumbent']]
            tmp2+=[True]
            incumbent = np.nan
            if tmp2.index(True)!=2:
                incumbent = tmp[tmp2.index(True)]
    rows.append((dt['state'], R, D, incumbent))
    
candidates = pd.DataFrame(rows, columns=["State", "R", "D", "incumbent"])
        
#remove second last name 
candidates.R = [candidate.split(' ')[-1] for candidate in candidates.R]
candidates

import datetime

def calc_sds(infoClean, polls):
    theo_sds = []
    obs_sds = []
    npolls = []
    for ii,election in enumerate(infoClean):
        #Note that the `polls` dictionary does not have a guaranteed ordering
        #because Python dictionaries are unordered object. For this reason, we
        #need to be careful and use the ordered `infoClean` object to make sure we
        #are correctly aligning the elements of `polls` and the rows in `candidates`.
        polls_key = election['slug']
        this_election = polls[polls_key]
            
        npoll = this_election.shape[0]

        #Use the candidates dataframe to find the name of the Republican in this race.
        #Use this to grab the column of polling results for this candidate.
        #Put poll results onto the proportion (not percentage) scale
        if candidates.R.ix[ii] in this_election.columns:
            p = this_election[candidates.R.ix[ii]]/100
            n = this_election["Number of Observations"]

            #Theoretical sd assumes a common value of p across all polls. Use the mean.
            p_mean = np.mean(p)
            theo_sd = np.sqrt(p_mean*(1-p_mean)*np.mean(1./n))

            #Observed sd is a simple calculation.
            obs_sd = np.std(p)

            theo_sds.append(theo_sd)
            obs_sds.append(obs_sd)
            npolls.append(npoll)

    return (theo_sds, obs_sds, npolls)

theo, obs, npolls = calc_sds(infoClean, polls)

#Grab the poll count from the infoClean object.
plt.scatter(theo, obs, marker="o", s=npolls*10)
plt.xlabel("Theoretical Standard Deviation")
plt.ylabel("Observed Standard Deviation")
plt.title("Observed vs Theoretical Polling SD's")

currentx = plt.xlim()
currenty = plt.ylim()
plt.plot((0,1),(0,1), c='black', linewidth=1)
plt.xlim(currentx)
plt.ylim(currenty)
plt.show()

#This time, do the scatterplot using text labels on each datapoint.
plt.scatter(theo, obs, marker="o", s=0)
plt.xlabel("Theoretical Standard Deviation")
plt.ylabel("Observed Standard Deviation")
plt.title("Observed vs Theoretical Polling SD's")

for i in range(len(theo)):
    plt.text(theo[i], obs[i], candidates.ix[i,'State'])

currentx = plt.xlim()
currenty = plt.ylim()
plt.plot((0,1),(0,1), c='black', linewidth=1)
plt.xlim(currentx)
plt.ylim(currenty)
plt.show()

# this is mean +- 2.58 SD also save this results to candidate df for 99% confidence interval
def calc_diffs(infoClean, polls):
    rows = []
    for ii,election in enumerate(infoClean):
        #Note that the `polls` dictionary does not have a guaranteed ordering
        #because Python dictionaries are unordered object. For this reason, we
        #need to be careful and use the ordered `infoClean` object to make sure we
        #are correctly aligning the elements of `polls` and the rows in `candidates`.
        polls_key = election['slug']
        this_election = polls[polls_key]
            
        npoll = this_election.shape[0]

        #Use the candidates dataframe to find the name of the Republican in this race.
        #Use this to grab the column of polling results for this candidate.
        #Put poll results onto the proportion (not percentage) scale
        if candidates.R.ix[ii] in this_election.columns:
            diffs = (this_election[candidates.R.ix[ii]]-this_election[candidates.D.ix[ii]])/100

            mean_diff = np.mean(diffs)
            obs_se = np.std(diffs)/np.sqrt(npoll)

            rows.append((polls_key, mean_diff, obs_se, mean_diff-obs_se*2.58, mean_diff+obs_se*2.58))

    return rows

ests = pd.DataFrame(calc_diffs(infoClean, polls), columns=['race', 'mean', 'se', 'lower', 'upper'])
print "Unsorted:"
print ests[['race', 'lower', 'upper']]
ests.sort("mean", inplace=True)
print "Sorted:"
print ests[['race', 'lower', 'upper']]

plt.errorbar(range(ests.shape[0]), ests['mean'], yerr=ests['se']*2.58, fmt='o')
plt.xticks(range(ests.shape[0]), ests['race'].values, rotation=90)
plt.xlim(-1, ests.shape[0])
plt.axhline(0, linewidth=1, color='black')
plt.xlabel("Race")
plt.ylabel("Difference")
plt.title("99% Confidence intervals for Diff (Rep-Dem) for each race")
plt.show()

# A cheap but slightly biased way to estimate tau is to take the
# standard deviation of the state polling means.
# Use `ests` from last question.
tau = np.std(ests['mean'])
B = (1/tau**2)/(1/ests['se']**2+1/tau**2)
ests['mu_post'] = (1-B)*ests['mean']

plt.scatter(ests['mean'], ests['mu_post'], s=50)
plt.title("Bayes vs. Raw Poll Averages")
plt.xlabel("Raw Average")
plt.ylabel("Bayes")

currentx = plt.xlim()
currenty = plt.ylim()
plt.plot((-1,1),(-1,1), c='black', linewidth=1)
plt.xlim(currentx)
plt.ylim(currenty)
plt.show()

#First, compute a standard deviation for each poll
ests['sd_post'] = np.sqrt((1-B)*ests['se']**2)

#Now, use normal CDF to find the posterior probability that the difference is greater than zero.
import scipy.stats
ests['R_win_prob'] = 1-scipy.stats.norm.cdf(0, loc=ests['mu_post'], scale=ests['sd_post'])
print ests[['race','R_win_prob']].sort('R_win_prob')

### made up numbers
num_states = ests.shape[0]

NSIM = 10000
simarr = np.zeros(NSIM, dtype=int)
for i in xrange(NSIM):
    simulated = 30 + np.sum(np.random.normal(ests['mu_post'], scale=ests['sd_post'], size =num_states) > 0 )
    simarr[i] = int(simulated)
plt.hist(simarr, bins=range(min(simarr)-1, max(simarr)+3))
plt.xlabel('Number of Seats in Republican Control')
plt.ylabel('Frequency')
plt.title('Monte Carlo simulation of Number of Seats in Republican Control')
plt.show()

