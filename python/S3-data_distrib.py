import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set_style('white')

colors = ['w']+['g' for g in range(19)]
main_counts = [i for i in [1, 25, 25, 7, 25, 25, 25, 6, 6, 11, 22, 5, 1, 5, 17, 7, 2, 6, 16, 3, 1, 25, 24, 1, 3, 32, 8, 25, 25, 25, 2, 25, 4] if i >= 5]
supplement_counts = [40,27,40,40,40,22,16,16,40,34,52,40,40,34,29]
fig, (main_ax, supp_ax) = plt.subplots(figsize=(7.5,3), nrows=1, ncols=2, facecolor='w')

sns.distplot(main_counts, color='g', bins=20, kde=False, ax=main_ax)
sns.distplot(supplement_counts, color='g', bins=20, kde=False, ax=supp_ax)

main_ax.set_xlabel('Main dataset:\nSequences per host', size=12)
supp_ax.set_xlabel('Supplemental dataset:\nSequences per host', size=12)
main_ax.set_ylabel('Number of hosts', size=12)
supp_ax.set_ylabel(None, visible=False, size=12)
main_ax.set_title('A', fontsize=18, ha='left', x=-0.12, y=1.08, fontweight='bold')
supp_ax.set_title('B', fontsize=18, ha='left', x=-0.12, y=1.08, fontweight='bold')

main_ax.set_xlim(0, 53)
supp_ax.set_xlim(0, 53)
main_ax.set_ylim(0, 12)
supp_ax.set_ylim(0, 12)

main_ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3, top="off", right="off")
supp_ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=3, top="off", right="off")

main_ax.spines['left'].set_linewidth(0.5)
main_ax.spines['bottom'].set_linewidth(0.5)
supp_ax.spines['left'].set_linewidth(0.5)
supp_ax.spines['bottom'].set_linewidth(0.5)
supp_ax.spines['top'].set_linewidth(0.5)
supp_ax.spines['right'].set_linewidth(0.5)

plt.tight_layout()
plt.savefig('../png/FigS3.png', bbox_inches='tight', dpi=600)



