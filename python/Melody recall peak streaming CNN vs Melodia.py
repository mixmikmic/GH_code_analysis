import motif
import motif.plot
import numpy as np
import mir_eval
import os
import medleydb as mdb
import seaborn
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

etr_cnn = motif.contour_extractors.DeepSal()
with open("../data_splits.json", 'r') as fhandle:
    dat_dict = json.load(fhandle)

all_scores_mel1 = []
all_scores_mel2 = []
all_scores_mel3 = []
trackids = []
for track_id in dat_dict['test']:
    npy_path = "../comparisons/multif0/experiment11b_output/fullmix_outputs/{}_prediction.npy".format(track_id)
    mtrack = mdb.MultiTrack(track_id)
    if os.path.exists(npy_path):
        print(track_id)
        trackids.append(track_id)
        ctr = etr_cnn.compute_contours(npy_path, mtrack.mix_path)

        scores_mel1 = ctr.coverage(mtrack.melody1_fpath, single_f0=True)
        scores_mel2 = ctr.coverage(mtrack.melody2_fpath, single_f0=True)
        scores_mel3 = ctr.coverage(mtrack.melody3_fpath, single_f0=False)
        print(scores_mel3['Recall'])
        
        all_scores_mel1.append(scores_mel1)
        all_scores_mel2.append(scores_mel2)
        all_scores_mel3.append(scores_mel3)

df_mel1 = pd.DataFrame(all_scores_mel1, index=trackids)
df_mel2 = pd.DataFrame(all_scores_mel2, index=trackids)
df_mel3 = pd.DataFrame(all_scores_mel3, index=trackids)

df_mel1.describe()

df_mel2.describe()

df_mel3.describe()

print(df_mel1['Recall'].mean())
print(df_mel2['Recall'].mean())
print(df_mel3['Recall'].mean())

df_mel1.to_csv("../outputs/deepsal_mel1_coverage_test.csv")
df_mel2.to_csv("../outputs/deepsal_mel2_coverage_test.csv")
df_mel3.to_csv("../outputs/deepsal_mel3_coverage_test.csv")

df_mel1 = pd.DataFrame.from_csv("deepsal_mel3_coverage_test.csv")
df_mel1.describe()

seaborn.set_style('white')
etr_cnn = motif.contour_extractors.DeepSal()
track_id = "MusicDelta_Gospel"
npy_path = "../experiment11b_output/fullmix_outputs/{}_prediction.npy".format(track_id)
mtrack = mdb.MultiTrack(track_id)
ctr = etr_cnn.compute_contours(npy_path, mtrack.mix_path)

sal = np.load(npy_path)
plt.figure(figsize=(30, 30))
plt.subplot(2, 1, 1)
motif.plot.plot_with_annotation(ctr, mtrack.melody2_fpath, single_f0=False)
plt.subplot(2, 1, 2)
plt.imshow(sal, origin='lower', cmap='hot')
plt.axis('auto')

# plt.savefig('/Users/rabitt/Desktop/swingjazz.pdf', format='pdf')

etr_sal = motif.contour_extractors.Salamon()
with open("../data_splits.json", 'r') as fhandle:
    dat_dict = json.load(fhandle)

all_scores_mel1_sal = []
all_scores_mel2_sal = []
all_scores_mel3_sal = []
trackids = []
for track_id in dat_dict['test']:
    npy_path = "../experiment11b_output/fullmix_outputs/{}_prediction.npy".format(track_id)
    mtrack = mdb.MultiTrack(track_id)
    if os.path.exists(npy_path):
        print(track_id)
        trackids.append(track_id)
        ctr = etr_cnn.compute_contours(mtrack.mix_path)

        scores_mel1 = ctr.coverage(mtrack.melody1_fpath, single_f0=True)
        scores_mel2 = ctr.coverage(mtrack.melody2_fpath, single_f0=True)
        scores_mel3 = ctr.coverage(mtrack.melody3_fpath, single_f0=False)
        print(scores_mel3['Recall'])
        
        all_scores_mel1_sal.append(scores_mel1)
        all_scores_mel2_sal.append(scores_mel2)
        all_scores_mel3_sal.append(scores_mel3)

df_mel1 = pd.DataFrame(all_scores_mel1_sal, index=trackids)
df_mel2 = pd.DataFrame(all_scores_mel2_sal, index=trackids)
df_mel3 = pd.DataFrame(all_scores_mel3_sal, index=trackids)

print(df_mel1['Recall'].mean())
print(df_mel2['Recall'].mean())
print(df_mel3['Recall'].mean())

df_mel1.to_csv("salamon_mel1_coverage_test.csv")
df_mel2.to_csv("salamon_mel2_coverage_test.csv")
df_mel3.to_csv("salamon_mel3_coverage_test.csv")

df_mel1 = pd.DataFrame.from_csv("salamon_mel3_coverage_test.csv")
df_mel1.describe()

