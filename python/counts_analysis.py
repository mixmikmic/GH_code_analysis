get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
plt.rcParams['image.cmap'] = 'RdBu'
import numpy as np
import scipy as sp
import scipy.stats as stats
import csv
import pandas

# load PubMed data & normalize by count
PM_counts_all = pandas.read_csv('../data/term_counts_pubmed.csv')
PM_counts = np.array(PM_counts_all.ix[1:, 2:])*1./np.array(PM_counts_all.ix[0, 2:])
terms = np.array(PM_counts_all.ix[1:,0])

# load CogSci data
CS_counts = np.array(pandas.read_csv('../data/term_counts_CS.csv', header=None))

# stack counts together
all_counts = np.append(CS_counts[1:]*1./CS_counts[0], PM_counts, axis=1)
columns = ['CogSci', 'PM Cogs', 'PM Neu', 'PM NeuMeth']

plt.figure(figsize=(10,10))
plt.loglog(all_counts[:,2], all_counts[:,0], '.', ms=10, alpha=0.9) #PMCog vs. PMNeu
plt.loglog(all_counts[:,2], all_counts[:,1], '.', ms=10, alpha=0.9)
plt.loglog(all_counts[:,2], all_counts[:,3], '.', ms=10, alpha=0.9)
plt.loglog([1e-5,1.], [1e-5,1.], '--k')
plt.xlim((1e-4, 0.25))
plt.ylim((1e-4, 0.25))

r1 = str(stats.pearsonr(all_counts[:,2],all_counts[:,0])[0])
r2 = str(stats.pearsonr(all_counts[:,2],all_counts[:,1])[0])
r3 = str(stats.pearsonr(all_counts[:,2],all_counts[:,3])[0])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(('PMNeu vs. CogSci, r='+ r1, 'PMNeu vs. PMCog, r='+r2, 'PMNeu vs. PMNeuMet, r='+r3), fontsize=20)

# term proportion difference between PMCog and PMNeuro
plt.figure(figsize=(10,8))

comps = [[0,2],[1,2]]
for i in range(2):
    plt.subplot(1,2,i+1)
    comp = comps[i]
    top_n = 15
    common_thresh = 0.005 #only include words that are in at least this proportion of entries in PM Cog
    common_idx = all_counts[:,1]>common_thresh
    common_counts = all_counts[common_idx,:]
    common_terms = terms[common_idx]
    print len(common_terms)
    termDiff = common_counts[:,comp[0]] - common_counts[:,comp[1]]


    plt.barh(range(top_n),termDiff[np.argsort(termDiff)][:top_n])
    plt.barh(range(top_n,top_n*2), termDiff[np.argsort(termDiff)][-top_n:])
    plt.title(columns[comp[0]] + '-' + columns[comp[1]], fontsize=15)
    plt.plot(plt.xlim(), [top_n-0.5]*2, 'r')
    print common_terms[np.argsort(termDiff)[:top_n]]
    print common_terms[np.argsort(termDiff)[:-(top_n+1):-1]]
    plt.yticks(range(top_n*2), np.append(common_terms[np.argsort(termDiff)[:top_n]],
        common_terms[np.argsort(termDiff)[-top_n:]]), rotation=0, fontsize=15);

plt.tight_layout()

plt.figure(figsize=(6,4))
for i in range(4):
    plt.semilogy(np.flip(np.sort(all_counts[:,i],), -1)[:250], lw=3)

plt.legend(columns,fontsize=12)
plt.xlabel('Term rank', fontsize=12)
plt.ylabel('Proportion of occurrence',fontsize=12);

top_N = 20
top_terms = []
top_counts = []
for i in range(4):
    idx = np.flip(np.argsort(all_counts[:,i],), -1)[:top_N]
    top_terms.append(terms[idx])
    top_counts.append(all_counts[idx, i])

with open('./data/topterms.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(top_N):
        line = []
        for db in range(4):
            line.append(top_terms[db][i])
            line.append("%.3f" % top_counts[db][i])
        print line
        writer.writerow(line)

# we can get unique words from each database
u_terms, u_count = np.unique(top_terms[0:4], return_counts=True)
for i in range(4):
    print columns[i]
    print np.intersect1d(top_terms[i], u_terms[u_count==1])
    print '---'

# or get words unique to neuro and cogs
neu_terms = np.unique(top_terms[2:])
cog_terms = np.unique(top_terms[:2])
u_terms, u_count = np.unique(np.append(neu_terms,cog_terms), return_counts=True)
for i in range(4):
    print columns[i]
    print np.intersect1d(top_terms[i], u_terms[u_count==1])
    print '---'



