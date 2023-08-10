import numpy as np
import mir_eval
import os
import medleydb as mdb
import seaborn
import glob
import json
import librosa
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

with open("../outputs/data_splits.json", 'r') as fhandle:
    dat_dict = json.load(fhandle)

all_mel_scores = []
for trackid in dat_dict['test']:
    print(trackid)
    mtrack = mdb.MultiTrack(trackid)
    
    pred_path = "../comparisons/melody2/juanjo_mdb_out/{}_mix.pitch".format(trackid)
    if not os.path.exists(pred_path) or not os.path.exists(mtrack.melody2_fpath):
        print(trackid)
        continue

    est_times, est_freqs = mir_eval.io.load_time_series(pred_path)

    mel2 = mtrack.melody2_annotation
    mel2 = np.array(mel2).T
    ref_times, ref_freqs = (mel2[0], mel2[1])
    
    plt.figure(figsize=(15, 7))
    plt.title(trackid)
    plt.plot(ref_times, ref_freqs, '.k', markersize=8)
    plt.plot(est_times, est_freqs, '.r', markersize=3)
    plt.show()

    mel_scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
    all_mel_scores.append(mel_scores)

mel_scores_df_partial = pd.DataFrame(all_mel_scores)
mel_scores_df_partial.to_csv("../outputs/juanjo_mdb_scores.csv")

mel_scores_df_partial.describe()

for trackid in dat_dict['test']:
    print(trackid)



