get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
from tqdm import tnrange, tqdm_notebook
sns.set_style("ticks")
import matplotlib
from matplotlib import rc
rc('text', usetex=False)
matplotlib.rcParams['svg.fonttype'] = 'none'
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

wd = '/projects/ps-yeolab3/bay001/maps/current/se_peak/'
# wd = '/projects/ps-yeolab3/bay001/maps/current/idr_peaks/'
output_dir = '/home/bay001/projects/gabe_qc_20170612/analysis/'
# peak_hist contains a file with the number of peaks intersecting a region and the number of peaks that don't.
# choose native cassette exons (all skipped exon regions)
# peak_hist = glob.glob(os.path.join(wd, '*native-cassette-exons.miso.hist.overlapping_peaks'))
peak_hist_excl = glob.glob(os.path.join(wd,'*excluded-upon-knockdown.hist.overlapping_peaks'))
peak_hist_incl = glob.glob(os.path.join(wd,'*included-upon-knockdown.hist.overlapping_peaks'))
len(peak_hist_excl)

peak_hist = peak_hist_incl
label='significantly included'

summed = 0 # sum of all peaks intersecting an event region
total_events = 0 # sum of all events combined
total_count = 0 # total number of RBPs counted.

all_percentages = defaultdict() # percent of intersecting/(intersecting+nonintersecting) peaks for each file
all_overlapping = defaultdict() # number of intersecting peaks for each file

for peak_hist in [peak_hist_incl, peak_hist_excl]:
    progress = tnrange(len(peak_hist))
    for fn in peak_hist:
        df = pd.read_table(fn)
        df['percentage'] = df['intersect'].div((df['intersect'] + df['no_intersect']))
        summed += df.ix[0]['intersect']
        total_events += (df.ix[0]['intersect'] + df.ix[0]['no_intersect'])
        total_count += 1
        all_overlapping[fn] = df['intersect']
        all_percentages[fn] = df['percentage']
        progress.update(1)

print("total number of peaks for all rbps: {}".format(summed))
print("total number of events for all rbps: {}".format(total_events))
print("total sample count: {}".format(total_count))
print("percentage of overlapping peaks: {}".format(summed/float(total_events)))

number_overlapping = pd.DataFrame(all_overlapping).T
number_overlapping.columns = ['Number']
number_overlapping['Number'].mean()

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.xaxis.set_visible(True)
number_overlapping['Number'].plot(kind='hist', bins=50, ax=ax)
plt.xlabel('Event/peaks overlaps')
plt.savefig(os.path.join(output_dir, 'overlapping_inputnormed_peaks.histogram.svg'))
output_dir

sns.swarmplot(number_overlapping['Number'])
plt.figsize = (1,1)
plt.title('Number {} events overlapping peaks'.format(label))
# plt.savefig(os.path.join(output_dir, '{}_overlapping_inputnormed_peaks.svg'.format(label.replace(' ','_'))))

percent_overlapping = pd.DataFrame(all_percentages).T
percent_overlapping = percent_overlapping*100
percent_overlapping.columns = ['Percent']

sns.swarmplot(percent_overlapping['Percent'])
plt.title('Percentage of {} events with overlapping peaks'.format(label))
# plt.savefig(os.path.join(output_dir, '{}_overlapping_inputnormed_peaks_percentage.svg'.format(label.replace(' ','_'))))

pd.read_table(peak_hist[1])

df = pd.read_table(
    pd.read_table(peak_hist[1])['infile'][0], 
    names=['chrom','start','end','p','f','strand']
)
df[(df['p']>3) & (df['f']>3)].shape

106+4293

from collections import defaultdict

peaks_intersecting = defaultdict(list)

progress = tnrange(len(peak_hist_excl)) # for each count of peak overlaps in excluded/kd sample:
for fn in peak_hist_excl:
    label = os.path.basename(fn).split('-excluded')[0] # just get the prefix (everything before -excluded-upon-knockdown)
    excl_df = pd.read_table(fn)
    if os.path.exists(fn.replace('excluded','included')): # if an 'included' file exists, read an append count
        incl_df = pd.read_table(fn.replace('excluded','included'))
        peaks_intersecting[label].append(incl_df.iloc[0]['intersect'])
    peaks_intersecting[label].append(excl_df.iloc[0]['intersect'])
    progress.update(1)
df = pd.DataFrame(peaks_intersecting).T
df.columns = ['Included','Excluded']
df.head()

df['Total'] = df.sum(axis=1)
df['log10Total'] = np.log10(df['Total'])
# df.to_csv(os.path.join(output_dir,'overlapping_peaks.txt'), sep='\t')

# sns.swarmplot(df['Total'], orient='v')
# plt.title('Significant events with overlapping peaks')
# plt.savefig(os.path.join(output_dir, 'overlapping_inputnormed_peaks.svg'))

# plt.title('Significant events with overlapping peaks (log10)')
# fig, ax = plt.subplots(111)
# 2.5 by 2.5
# dots smaller (3 or 4)
# font size 8?
# 
# ax = sns.swarmplot(df['log10Total'], orient='v')
# plt.savefig(os.path.join(output_dir, 'overlapping_inputnormed_peaks.log10.svg'))

### Don't use this - this plots the total overlap (incl + excl), whereas we really want incl, excl separately (they plot separate lines)
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.xaxis.set_visible(True)
# df['Total'].plot(kind='hist', bins=50, ax=ax)
# plt.xlabel('Event/peaks overlaps')
# plt.savefig(os.path.join(output_dir, 'overlapping_inputnormed_peaks.histogram.svg'))
# output_dir

df['Total'].median()

df['Total'].mean()

df.sort_values(by='Total', ascending=True)

df = pd.read_table(
    '/home/elvannostrand/data/clip/CLIPseq_analysis/ENCODE_FINALforpapers_20170325/507_02.basedon_507_02.peaks.l2inputnormnew.bed.compressed.bed',
    names=['chrom','start','end','name','score','strand']
)
df = df[(df['name']>=3) & (df['score']>=3)]
df.shape





