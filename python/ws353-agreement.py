from common_imports import *
from skll.metrics import kappa
from scipy.stats import spearmanr
from itertools import combinations

sns.timeseries.algo.bootstrap = my_bootstrap
sns.categorical.bootstrap = my_bootstrap

columns = 'Word 1,Word 2,Human (mean),1,2,3,4,5,6,7,8,9,10,11,12,13'.split(',')
df1 = pd.read_csv('similarity-data/wordsim353/set1.csv')[columns]
df2 = pd.read_csv('similarity-data/wordsim353/set2.csv')[columns]
df = pd.concat([df1, df2], ignore_index=True)
df_gold = pd.read_csv('similarity-data/wordsim353/combined.csv',
                     names='w1 w2 sim'.split())

# had to remove trailing space from their files to make it parse with pandas
marco = pd.read_csv('similarity-data/MEN/agreement/marcos-men-ratings.txt',
                   sep='\t', index_col=[0,1], names=['w1', 'w2', 'sim']).sort_index().convert_objects(convert_numeric=True)
elia = pd.read_csv('similarity-data/MEN/agreement/elias-men-ratings.txt',
                   sep='\t', index_col=[0,1], names=['w1', 'w2', 'sim']).sort_index().convert_objects(convert_numeric=True)

df.head()

# Each index ``i`` returned is such that ``bins[i-1] <= x < bins[i]``
def bin(arr, nbins=2, debug=False):
    bins = np.linspace(arr.min(), arr.max(), nbins+1)
    if debug:
        print('bins are', bins)
    return np.digitize(arr, bins[1:-1])

bin(df['1'], nbins=5, debug=True)[:10]

bin(np.array([0, 2.1, 5.8, 7.9, 10]), debug=True) # 0 and 10 are needed to define the range of values

bin(np.array([0, 2.1, 5.8, 7.9, 10]), nbins=3, debug=True)

df.describe()

elia.describe()

bin_counts = range(2, 6)
# pair, bin count, kappa
kappas_pair = []
for name1, name2 in combinations(range(1,14), 2):
    for b in bin_counts:
        kappas_pair.append(['%d-%d'%(name1, name2), 
                       b, 
                       kappa(bin(df[str(name1)], b), bin(df[str(name2)], b))])

kappas_mean = []
for name in range(1, 14):
    for b in bin_counts:
        kappas_mean.append(['%d-m'%name, 
                       b, 
                       kappa(bin(df[str(name)], b), bin(df_gold.sim, b))])

kappas_men = [] # MEN data set- marco vs elia
for b in bin_counts:
    kappas_men.append(['marco-elia',
                       b,
                       kappa(bin(marco.sim.values, b), bin(elia.sim.values, b))])

kappas1 = pd.DataFrame(kappas_pair, columns=['pair', 'bins', 'kappa'])
kappas1['kind'] = 'WS353-P'
kappas2 = pd.DataFrame(kappas_mean, columns=['pair', 'bins', 'kappa'])
kappas2['kind'] = 'WS353-M'
kappas3 = pd.DataFrame(kappas_men, columns=['pair', 'bins', 'kappa'])
kappas3['kind'] = 'MEN'
kappas = pd.concat([kappas1, kappas2, kappas3], ignore_index=True)
kappas.head(3)

with sns.color_palette("cubehelix", 3):
    ax = sns.tsplot(kappas, time='bins', unit='pair', condition='kind', value='kappa', 
               marker='s', linewidth=4);

sparsify_axis_labels_old(ax)
ax.set_xlabel('Bins')
ax.set_ylabel('Kohen $\kappa$')
ax.set_xticklabels([2, 2, 3, 3, 4, 4, 5, 5])
sns.despine()
plt.savefig('ws353-kappas.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

kappas.groupby(['bins', 'kind']).mean()

rhos_pair = []
for name1, name2 in combinations(range(1,14), 2):
    rhos_pair.append(spearmanr(bin(df[str(name1)], b), bin(df[str(name2)], b))[0])
    
rhos_mean = []
for name in range(1,14):
    rhos_mean.append(spearmanr(bin(df[str(name)], b), bin(df_gold.sim, b))[0])

sns.distplot(rhos_pair, label='pairwise');
# plt.axvline(np.mean(rhos_pair));

sns.distplot(rhos_mean, label='to mean');
# plt.axvline(np.mean(rhos_mean), color='g');
plt.legend(loc='upper left');
plt.savefig('ws353-rhos.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
print(np.mean(rhos_pair), np.mean(rhos_mean))

spearmanr(marco.sim, elia.sim) # they report .6845

men = pd.DataFrame({'marco':marco.sim.values, 'elia':elia.sim.values})
sns.jointplot(x='marco', y='elia', data=men, kind='kde', space=0).set_axis_labels('Judge 1', 'Judge 2')
plt.savefig('jointplot-men.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

# difference can reach 6 points, and 10% of all data is more than 2 points aways
(men.marco - men.elia).abs().value_counts().cumsum()

men.marco.describe()



