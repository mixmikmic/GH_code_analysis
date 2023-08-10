get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tnrange, tqdm_notebook

colors = sns.color_palette("hls", 8)

p1 = '/projects/ps-yeolab3/bay001/tbos/clip_analysis_output/A1_01_01.basedon_A1_01_01.peaks.l2inputnormnew.bed.compressed.p3.fc3.bed.clip_analysis.pickle'
p2 = '/projects/ps-yeolab3/bay001/tbos/clip_analysis_output/A1_02_01.basedon_A1_02_01.peaks.l2inputnormnew.bed.compressed.p3.fc3.bed.clip_analysis.pickle'

loaded1 = pickle.load(open(p1, 'rb'))
loaded2 = pickle.load(open(p2, 'rb'))

# look at all the info inside the pickle file
loaded1.keys()

# access kmer counts:
loaded1['kmer_results'].keys()

kmer_len = 6
moi = ['TAATGT','AAATGT','AATGTA','AATGTG','AATGTT','ATGTAT'] # we can highlight specific kmers

def plot_zscore_enrichment(loaded1, loaded2, moi=[]):
    motif_scores1 = {}
    motif_scores2 = {}
    for key, value in loaded1['kmer_results']['all'][kmer_len].iteritems():
        motif_scores1[key] = value.delta
        motif_scores2[key] = loaded2['kmer_results']['all'][kmer_len][key].delta

    join1 = pd.DataFrame(motif_scores1,index=['rep1']).T
    join2 = pd.DataFrame(motif_scores2,index=['rep2']).T
    merged = pd.merge(join1, join2, how='left', left_index=True, right_index=True)
    x = plt.scatter(merged['rep1'],merged['rep2'])
    
    for m in moi:
        x = plt.scatter(merged['rep1'].ix[m], merged['rep2'].ix[m], color='red')
        # print(m, merged['rep1'].ix[m], merged['rep2'].ix[m])
        
plot_zscore_enrichment(loaded1, loaded2, moi)

p1 = '/projects/ps-yeolab3/encode/analysis/encode_idr_clip_analysis/203.01v02.IDR.out.0102merged.bed.annotated.clip_formatted.bed.clip_analysis.pickle'
loaded1 = pickle.load(open(p1, 'rb'))

loaded1['region_sizes']

labels = 'cds', 'distintron', '5utrs', 'proxintron', '3utrs' # , 'na'
sizes = [
    loaded1['region_sizes']['cds'], 
    loaded1['region_sizes']['distintron500'], 
    loaded1['region_sizes']['five_prime_utrs'], 
    loaded1['region_sizes']['proxintron500'], 
    loaded1['region_sizes']['three_prime_utrs'], 
    # loaded1['region_sizes']['uncatagorized'], 
]

explode = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



