get_ipython().magic('qtconsole')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('load_ext ipycache')

import sys
sys.path.append('/home/jfear/devel/GalaxyTools')

from baPlot import *
get_ipython().magic('matplotlib inline')

myopts = ['--input', '/home/jfear/sandbox/secim/data/ST000015_log.tsv',
          '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
          '--ID', 'Name',
          '--group', 'treatment',
          '--ba', '/home/jfear/sandbox/secim/data/test_ba.pdf',
          '--flag_dist', '/home/jfear/sandbox/secim/data/test_dist.pdf',
          '--flag_summary', '/home/jfear/sandbox/secim/data/test_flag_summary.tsv',
          '--debug']

args = getOptions(myopts=myopts)

dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)

if args.processOnly:
    toProcess = dat.design[dat.design[args.group].isin(args.processOnly)].index
else:
    toProcess = dat.sampleIDs
    
wide = dat.wide.loc[:, toProcess]

# Create a FlagOutlier object to store all flags
flags = FlagOutlier(dat.wide.index)

# Open a multiple page PDF for plots
ppBA = PdfPages(args.baName)

# Grab group
grp = dat.design.groupby(dat.group)
i = '09_uM_palmita'
val = grp.get_group(i)
#val = dat.design

# Create combos
combos = list(combinations(val.index, 2))

get_ipython().run_cell_magic('cache', 'ba_flags.pkl flags', '\niterateCombo(wide, combos, ppBA, flags, args.cutoff, group=i)')

combo = combos[0]
data = wide
out = ppBA
cutoff = 3

rows = data.shape[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
fig.subplots_adjust(wspace=0.4)
subset = data.loc[:, [combo[0], combo[1]]]

# Drop missing value
subset.dropna(inplace=True)
missing = rows - subset.shape[0]

# Set X's and Y's
x, y = subset.iloc[:, 0], subset.iloc[:, 1]
x.name = 'sample1'
y.name = 'sample2'

# Scatter Plot
ax1 = makeScatter(x, y, ax1)

# BA plot
ax2, outlier = makeBA(x, y, ax2, cutoff)
fig.savefig('/home/jfear/devel/GalaxyTools/images/ba_plot.png', bbox_inches='tight')

summary = flags.summarizeSampleFlags(wide)

# Sum flags across samples
col_sum = summary.sum(axis=0)
col_sum.sort(ascending=False)

# Sum flags across compounds
row_sum = summary.sum(axis=1)
row_sum.sort(ascending=False)

# How many flags could I have
row_max, col_max = summary.shape

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

col_sum[col_sum > 0].plot(kind='bar', ax=ax1)
row_sum[row_sum > 0].head(20).plot(kind='bar', ax=ax2)
ax1.set_xlabel('samples'); ax2.set_xlabel('compounds')
ax1.set_ylabel('Num Outliers'); ax2.set_ylabel('Num Outliers')
fig.savefig('/home/jfear/devel/GalaxyTools/images/ba_plot_outlier_dist.png', bbox_inches='tight')



