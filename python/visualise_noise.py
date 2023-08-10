from common_imports import *

sns.timeseries.algo.bootstrap = my_bootstrap

df = pd.read_csv('intrinsic_noise_word_level.csv', index_col=0)
df['Dataset'] = df.test.map(str.upper)
rand_df = df[df.vect == 'random']
df = df[(df.vect == 'wtv-wiki-100') & (df.kind == 'strict')]
# df = df.drop(['test', 'vect'], axis=1)
noise_df = df
ORDER = ['MEN', 'SIMLEX', 'WS353', 'RG', 'MC']

SIZES = {'MEN':3000, 'SIMLEX':999, 'WS353':353, 'RG':65, 'MC':30}
df.head()

def fifth_perc(x):
    return np.percentile(x, 2.5)

def ninetieth_perc(x):
    return np.percentile(x, 97.5)

(df[df.noise == 0]
  .groupby('Dataset')
  .agg([np.mean, np.std, np.min, fifth_perc, ninetieth_perc, np.max])[['corr']]
  .reset_index()
  .set_index('Dataset'))

(rand_df[rand_df.noise == 0]
      .groupby('Dataset')
      .agg([np.mean, np.std, np.min, fifth_perc, ninetieth_perc, np.max])[['corr']]
      .reset_index()
      .set_index('Dataset'))

# not enough space, split in two subfigures
groups = {'MEN': 'A',
          'RG': 'A',
          'MC':'A',
          'SIMLEX': 'B',
          'WS353': 'B', }
df['group'] = df.Dataset.map(lambda x: groups[x])
df.head()

sns.set(font_scale=1.1)
sns.set_style("white")
for i, subdf in df.groupby('group'):
    sns.set_palette("cubehelix", 1)
    plt.figure(figsize=(6,4))
    ax = sns.tsplot(time='noise', value='corr', condition='Dataset', 
                unit='folds', ci=68, data=subdf)

    ax.axhline(0, c='k', linestyle='dashed')
    
    subset = rand_df[(rand_df.noise == 0) & (rand_df.Dataset == 'MC')]
#     ax.fill_between(rand_df.noise.unique(),
#                    fifth_perc(subset['corr']), 
#                    ninetieth_perc(subset['corr']),
#                    color='grey', alpha=0.5)

    sparsify_axis_labels(ax, x=1, y=.5)
    ax.set_ylabel('Spearman $\\rho$')
    ax.set_xlabel('Noise parameter (n)')
    sns.despine(ax=ax)#, bottom=True, left=True)
    plt.savefig('bootstrapped-corr-%s.pdf'%i, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

