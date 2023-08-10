import skbio
import numpy as np
import qiime_default_reference
from q2_alignment._filter import _compute_conservation_mask, _compute_frequencies

lane_mask = [i == '1' for i in qiime_default_reference.get_template_alignment_column_mask().decode()]
print(qiime_default_reference.__version__)

def compare_masks(mask1, mask2):
    if len(mask1) != len(mask2):
        raise ValueError('Only masks of equal length can be compared.')
    matches = sum(np.array(mask1) == np.array(mask2))
    return matches / len(mask1)

def compute_and_compare_masks(alignment_path, lane_mask):
    alignment = skbio.TabularMSA.read(alignment_path, constructor=skbio.DNA, lowercase=True)
    frequencies = _compute_frequencies(alignment)
    result = {}
    for c in np.arange(0.0, 1.01, 0.05):
        mask = _compute_conservation_mask(frequencies, skbio.DNA, c)
        result[c] = compare_masks(mask, lane_mask)
    return result

import glob
import os
alignment_paths = glob.glob("/Users/caporaso/data/gg_13_8_otus/rep_set_aligned/*_otus.fasta")

data = {}
for alignment_path in alignment_paths:
    percent_id = int(os.path.split(alignment_path)[1][:2])
    data[percent_id] = compute_and_compare_masks(alignment_path, lane_mask)

import pandas as pd
df = pd.DataFrame(data)

get_ipython().magic('matplotlib inline')

import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig.set_size_inches((12, 12))
sns.heatmap(df, vmin=0.80, vmax=1.0, annot=True, ax=ax)
ax.set_xlabel('Reference OTU percent identity')
ax.set_ylabel('Minimum conservation threshold')
ax.set_title('Fraction of identical positions between dynamically computed mask and Lane mask')

df.to_csv('mask_match_summary.csv')

df.corr()



