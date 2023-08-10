import vamp
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

def get_melodia_output(audio_fpath, thresh):
    y, fs = librosa.load(audio_fpath, sr=None)
    output = vamp.collect(
        y, fs, 'mtg-melodia:melodia', output='melody',
        parameters={'voicing': thresh}
    )
    hop = float(output['vector'][0])
    pitch = np.array(output['vector'][1])
    times = np.arange(0, hop*len(pitch), hop)
    return times, pitch

thresh_vals = np.arange(0, 1, 0.1)
mel_accuracy = {v: [] for v in thresh_vals}

for trackid in dat_dict['validate']:

    mtrack = mdb.MultiTrack(trackid)
    if mtrack.dataset_version != 'V1':
        continue
    
    print(trackid)
    mel2 = mtrack.melody2_annotation
    mel2 = np.array(mel2).T
    ref_times, ref_freqs = (mel2[0], mel2[1])

    for thresh in thresh_vals:
        est_times, est_freqs = get_melodia_output(mtrack.mix_path, thresh)
        mel_scores = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        mel_accuracy[thresh].append(mel_scores['Overall Accuracy'])

accuracy_vals = [np.mean(mel_accuracy[thresh]) for thresh in thresh_vals]
best_thresh_idx = np.argmax(accuracy_vals)
best_thresh = thresh_vals[best_thresh_idx]

print("Best threshold is {} with an OA of {}".format(
    best_thresh, accuracy_vals[best_thresh_idx])
)

print accuracy_vals

all_mel_scores = []
for trackid in dat_dict['test']:
    print(trackid)
    mtrack = mdb.MultiTrack(trackid)
    
    if not os.path.exists(mtrack.melody2_fpath):
        print(trackid)
        continue
    
    est_times, est_freqs = get_melodia_output(mtrack.mix_path, best_thresh)

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
mel_scores_df_partial.to_csv("../outputs/Melodia_scores.csv")

mel_scores_df_partial.describe()



