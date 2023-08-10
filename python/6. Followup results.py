get_ipython().magic('pylab inline')
import pandas as pd
import seaborn as sns

annotation = pd.read_csv('../zv9_gene_annotation.txt', index_col=0)

sample_info = pd.read_csv('sample_info.csv', index_col=0)

tpm = pd.read_csv('followup_tpm.csv', index_col=0)
ercc_idx = filter(lambda s: 'ERCC-' in s, tpm.index)

etpm = tpm.drop(ercc_idx)
etpm = etpm.drop('GFP')

etpm = etpm / etpm.sum() * 1e6

# Original data
original_sample_data = pd.read_csv('../sample_info_qc_pt.csv', index_col=0).sort_index(0).sort_index(1)
original_sample_data = original_sample_data.ix[original_sample_data["Pass QC"]]
original_sample_data = original_sample_data.query('cluster != "x"')

from ast import literal_eval
original_sample_data['cluster_color'] = original_sample_data['cluster_color'].apply(literal_eval)

import statsmodels.formula.api as smf 

sample_info = sample_info.query('pass_qc')

sample_info.Population.value_counts()

sample_info.groupby('plate').first()

sample_info['location'] = sample_info.plate.map(lambda p: 'Circulation' if p == 3 else 'Kidney')

sample_info['Population'].map({'P5': 'EarlyEnriched', 'low': 'EGFP-low', 'high': 'EGFP-high'})

sample_info['condition'] = sample_info['location'] + ' ' + sample_info['Population']

sample_info['condition'] = sample_info['location'] + '\n' + sample_info['Population'].map({'P5': 'EarlyEnriched', 'low': 'EGFP-low', 'high': 'EGFP-high'})

sample_info['condition'].value_counts()

ordering = sample_info.condition.sort(inplace=False).unique()[[4, 3, 2, 1, 0]]

sns.set_style('whitegrid');

sns.stripplot(x='condition',
              y='num_genes',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');

plt.ylabel('Number of expressed genes');
plt.title('Number of expressed genes');
plt.savefig('../cell-reports-rebuttal/followup_expressed_genes.pdf');

ccol = original_sample_data.groupby('cluster')['cluster_color'].first()

sns.stripplot(x='cluster',
              y='detected_genes',
              data=original_sample_data,
              jitter=True,
              order=['1a', '1b', '2', '3', '4'],
              palette=ccol);

plt.ylabel('Number of expressed genes');
plt.title('Number of expressed genes');

jitter_width = 0.25

column = 'detected_genes'
y = original_sample_data.query('cluster in ["1a", "1b", "2"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 1, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["1a", "1b", "2", "3"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 2, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["4"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 3, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

plt.xlim(0, 5);

plt.xticks([1, 2, 3], ['EarlyEnriched\n(1a, 1b, 2)', 'EGFP-low\n(1a, 1b, 2, 3)', 'EGFP-high\n(4)']);
plt.title('Number of expressed genes');
sns.axlabel('Equivalent conditions in original Kidney experiment', 'Number of expressed genes');
plt.savefig('../cell-reports-rebuttal/followup_expressed_genes-original.pdf');



sample_info['mRNA_content'] = 1e6 - sample_info['ERCC_content']

sns.stripplot(x='condition',
              y='mRNA_content',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');

plt.ylabel('Endogenous mRNA Content\n(TPM)');
plt.title('Endogenous mRNA Content');
plt.savefig('../cell-reports-rebuttal/followup_rna_content.pdf');

sns.stripplot(x='cluster',
              y='mRNA_content',
              data=original_sample_data,
              jitter=True,
              order=['1a', '1b', '2', '3', '4'],
              palette=ccol);

plt.ylabel('Endogenous mRNA Content\n(TPM)');
plt.title('Endogenous mRNA Content');

jitter_width = 0.25

column = 'mRNA_content'
y = original_sample_data.query('cluster in ["1a", "1b", "2"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 1, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["1a", "1b", "2", "3"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 2, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["4"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 3, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

plt.xlim(0, 5);

plt.xticks([1, 2, 3], ['EarlyEnriched\n(1a, 1b, 2)', 'EGFP-low\n(1a, 1b, 2, 3)', 'EGFP-high\n(4)']);
sns.axlabel('Equivalent conditions in original Kidney experiment', 'Endogenous mRNA Content\n(TPM)');
plt.title('Endogenous mRNA Content');
plt.savefig('../cell-reports-rebuttal/followup_rna_content-original.pdf');

tmp = sample_info.copy()
tmp.GFP += 1
sns.stripplot(x='condition',
              y='GFP',
              data=tmp,
              jitter=True,
              order=ordering,
              color='k');

sns.axlabel('', 'EGFP mRNA Expression (TPM)');
plt.title('EGFP mRNA Expression');

plt.yscale('log');
plt.savefig('../figures/followup_egfp_expr.pdf');



full_res = smf.ols('np.log10(GFP) ~ 1 + C(location)', data=tmp.query('Population == "high"')).fit()
res_res = smf.ols('np.log10(GFP) ~ 1', data=tmp.query('Population == "high"')).fit()
full_res.compare_lr_test(res_res)

sns.stripplot(x='condition',
              y='488',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');

sns.axlabel('', 'EGFP Flourescence');
plt.title('EGFP Flourescence');

plt.yscale('log');

sample_info['log_488'] = np.log10(sample_info['488'])

full_res = smf.ols('log_488 ~ 1 + C(location)', data=sample_info.query('Population == "high"')).fit()
res_res = smf.ols('log_488 ~ 1', data=sample_info.query('Population == "high"')).fit()
full_res.compare_lr_test(res_res)

sample_info['CD41'] = etpm.ix['ENSDARG00000018687'] + 1
sns.stripplot(x='condition',
              y='CD41',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');
plt.yscale('log');
plt.savefig('../cell-reports-rebuttal/followup_cd41_expression.pdf');



original_genes = pd.read_csv('../gene_expression_s.csv', index_col=0).sort_index(0).sort_index(1)
original_egenes = original_genes.drop(ercc_idx)
original_egenes = original_egenes.drop('GFP')
original_egenes = (original_egenes / original_egenes.sum()) * 1e6

original_sample_data['CD41'] = original_egenes.ix['ENSDARG00000018687'] + 1
sns.stripplot(x='cluster',
              y='CD41',
              data=original_sample_data,
              jitter=True,
              order=['1a', '1b', '2', '3', '4'],
              palette=ccol);

plt.yscale('log')

jitter_width = 0.25

column = 'CD41'
y = original_sample_data.query('cluster in ["1a", "1b", "2"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 1, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["1a", "1b", "2", "3"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 2, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

y = original_sample_data.query('cluster in ["4"]')[column]
plt.scatter(np.random.rand(y.shape[0]) * jitter_width - jitter_width / 2 + 3, y,
            color=original_sample_data.ix[y.index, 'cluster_color'],
            s=50,
            edgecolor='w',
            lw=1);

plt.yscale('log');

plt.xlim(0, 5);

plt.xticks([1, 2, 3], ['EarlyEnriched\n(1a, 1b, 2)', 'EGFP-low\n(1a, 1b, 2, 3)', 'EGFP-high\n(4)']);
sns.axlabel('Equivalent conditions in original Kidney experiment', 'CD41 (TPM)');
plt.savefig('../cell-reports-rebuttal/followup_cd41_expression-original.pdf');



full_res = smf.ols('CD41 ~ 1 + C(location)', data=sample_info.query('Population == "high"')).fit()
res_res = smf.ols('CD41 ~ 1', data=sample_info.query('Population == "high"')).fit()
full_res.compare_lr_test(res_res)

sample_info.Population.value_counts()

full_res = smf.ols('CD41 ~ 1 + C(Population)', data=sample_info.query('Population in ["high", "low"] & location == "Kidney"')).fit()
res_res = smf.ols('CD41 ~ 1', data=sample_info.query('Population in ["high", "low"] & location == "Kidney"')).fit()
full_res.compare_lr_test(res_res)

full_res = smf.ols('CD41 ~ 1 + C(Population)', data=sample_info.query('Population in ["P5", "low"] & location == "Kidney"')).fit()
res_res = smf.ols('CD41 ~ 1', data=sample_info.query('Population in ["P5", "low"] & location == "Kidney"')).fit()
full_res.compare_lr_test(res_res)



sample_info['vwf'] = etpm.ix['ENSDARG00000077231'] + 1
sns.stripplot(x='condition',
              y='vwf',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');
plt.yscale('log');

exp_etpm = etpm[((etpm > 1).sum(1) > 2)]



kidney_idx = sample_info.query('Population == "high" & location == "Kidney"').index
circulation_idx = sample_info.query('Population == "high" & location == "Circulation"').index



from scipy import stats

annotation = pd.read_csv('../zv9_gene_annotation.txt', index_col=0, sep='\t')

# Perform LR test for every gene

tmp = sample_info.query('Population == "high"').copy()

test_results = pd.Series(index=exp_etpm.index)
for gene in exp_etpm.index:
    tmp['tmp_gene'] = np.log10(etpm.ix[gene, tmp.index] + 1)
    full_res = smf.ols('tmp_gene ~ 1 + C(location)', data=tmp).fit()
    res_res = smf.ols('tmp_gene ~ 1', data=tmp).fit()
    test_results[gene] = full_res.compare_lr_test(res_res)[1]

test_results.dropna().shape

sns.kdeplot(test_results, bw=0.005);

test_results.hist(bins=32);

test_results.mode()

test_results[np.isclose(test_results, 0.4347)].head()



sample_info['tmp_gene'] = etpm.ix['ENSDARG00000043137'] + 1
sns.stripplot(x='condition',
              y='tmp_gene',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');
plt.yscale('log');

# How many genes _can_ we assess?

idx = sample_info.query('Population == "high"').index
((etpm[idx] > 1).sum(1) > 2).sum()



test_results[(test_results < 0.005)].sort(inplace=False)

annotation.ix[test_results[(test_results < 0.005)].sort(inplace=False).index]

sample_info['tmp_gene'] = etpm.ix['ENSDARG00000076128'] + 1
sns.stripplot(x='condition',
              y='tmp_gene',
              data=sample_info,
              jitter=True,
              order=ordering,
              color='k');
plt.yscale('log');



plt.scatter(exp_etpm[kidney_idx].median(1) + 1, exp_etpm[circulation_idx].median(1) + 1,c=test_results.replace(np.nan, 1));
plt.loglog();



from statsmodels.sandbox.stats.multicomp import multipletests 

apval = multipletests(test_results)[1]

apval = pd.Series(apval, index=test_results.index)

apval.hist();

apval.value_counts()

apval.dropna().sort(inplace=False).head()

annotation.ix['ENSDARG00000076128']



