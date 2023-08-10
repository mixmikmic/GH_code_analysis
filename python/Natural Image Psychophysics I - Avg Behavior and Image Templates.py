import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
get_ipython().run_line_magic('matplotlib', 'inline')

from swdb2017.brain_observatory import psychophysics

df = psychophysics.get_natural_image_psychophysics_df()
df

r = df.iloc[:,1:].values

fig,ax = plt.subplots(1,1)
im = ax.imshow(r,cmap='magma')
plt.colorbar(im,ax=ax,label='Response probability')
ax.set_ylabel('Pre-change image')
ax.set_xlabel('Post-change image')
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(df.change_image.values+1)
ax.set_yticklabels(df.change_image.values+1)
ax.set_title('Avg Change Detection Behavior (n=10 mice)')

fig,ax = plt.subplots(1,1,figsize=(5,5))
avg = r.mean(axis=0) # compute avg detectability
idx = np.argsort(avg) # get index for sorting by avg resp probability
ax.plot(avg[idx],'-o',label='Change')
ax.plot(np.diagonal(r)[idx],'-o',label='No change')
ax.set_xticks(range(8))
ax.set_xticklabels(df.change_image.values[idx]+1)
ax.set_ylim(0,1)
ax.set_ylabel('Behavioral response probability')
ax.set_xlabel('Natural image index')
ax.legend()

fig,ax = plt.subplots(1,1,figsize=(5,5))
ax.plot(r.mean(axis=0),'-o',label='Change')
ax.plot(np.diagonal(r),'-o',label='No change')
ax.set_xticks(range(8))
ax.set_xticklabels(df.change_image.values+1)
ax.set_ylim(0,1)
ax.set_ylabel('Behavioral response probability')
ax.set_xlabel('Natural image index')
ax.legend()

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'
manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

expts = pd.DataFrame(boc.get_ophys_experiments(stimuli=['natural_scenes']))
data_set = boc.get_ophys_experiment_data(expts.id[0])
ni_templates = data_set.get_stimulus_template(stimulus_name='natural_scenes')

fig,ax = plt.subplots(2,4,figsize=(11,5))
ax = ax.ravel()
for i,stim_ind in enumerate(df.change_image.values):
    ax[i].imshow(ni_templates[stim_ind,:,:],cmap='gray')
    ax[i].set_title('image: %s'%str(stim_ind+1)) # add 1 to index to match brain-map.org
    ax[i].axis('off')

