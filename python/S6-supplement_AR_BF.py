import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd

actualRates = pd.read_csv('../../beast/supplement/discreteTraits/results/discreteTraits_2016-12-27_actualRates.csv', index_col=0, header=0)
bayesFactors = pd.read_csv('../../beast/supplement/discreteTraits/results/discreteTraits_2016-12-27_bf.csv', index_col=0, header=0)

rename = { i: i.replace('_', ' ') for i in actualRates.columns.values }
actualRates.rename(columns = rename, index = rename, inplace=True)
bayesFactors.rename(columns = rename, index = rename, inplace=True)
bayesFactors = bayesFactors.applymap(lambda x: x / 12.0)

fig, (ar_ax, bf_ax) = plt.subplots(figsize=(5.2, 7.5),nrows=2,ncols=1,facecolor='w')

ar_heatmap = sns.heatmap(actualRates, cmap = plt.cm.GnBu, square = True, #annot=True,
                         linecolor = 'black', robust=True, linewidths = 0.1, ax=ar_ax)

bf_heatmap = sns.heatmap(bayesFactors, cmap = plt.cm.GnBu, square = True, #annot=True,
                         linecolor = 'black', robust=True, linewidths = 0.1, ax=bf_ax, vmin=3.0)

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=8)
cax2=plt.gcf().axes[2]
cax2.tick_params(labelsize=8)
for label in ar_ax.get_yticklabels() + ar_ax.get_xticklabels() + bf_ax.get_xticklabels() + bf_ax.get_yticklabels():
        label.set_size(8)

ar_ax.set_title('Actual Rates:\nRates * Indicators', fontsize=12)
ar_ax.set_ylabel('From Host 1', fontsize=12, va='bottom',fontweight='bold')
ar_ax.set_xlabel('To Host 2', fontsize=12, va='top',fontweight='bold')
bf_ax.set_title('Bayes factors:\nEvidence for Non-Zero Actual Rate', fontsize=12)

bf_ax.set_ylabel('From Host 1', fontsize=12, va='bottom',fontweight='bold')
bf_ax.set_xlabel('To Host 2', fontsize=12, va='top',fontweight='bold')

ar_ax.text(-15, 15, 'A', fontsize=18, fontweight='bold')
bf_ax.text(-15,15, 'B', fontsize=18, fontweight='bold')

plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../png/FigS6.png', dpi=300, bbox_inches='tight')
plt.show()



