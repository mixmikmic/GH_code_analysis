import sys
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/celiaberon/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as scio
import bandit_preprocessing as bp
import sys
import os
import matplotlib.pyplot as plt
import calcium_codes as cc
import hmm_on_behavior as hob
get_ipython().magic('matplotlib inline')

record = pd.read_csv('/Users/celiaberon/GitHub/mouse_bandit/session_record.csv',index_col=0)
record_path = '/Users/celiaberon/GitHub/mouse_bandit/session_record.csv'
ca_data = scio.loadmat('/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat',squeeze_me = True, struct_as_record = False)
#ca_data = scio.loadmat('/Volumes/Neurobio/MICROSCOPE/Celia/data/q43_03202017_bandit_8020/q43_03202017_neuron_master.mat',squeeze_me = True, struct_as_record = False)
neuron = ca_data['neuron_results'] 

record.head(2)

session_name  = '03142017_K7'
mouse_id = 'K7'

record[record['Session ID'] == session_name]

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked',
           'Right Reward Prob','Left Reward Prob','Reward Given',
          'center_frame','decision_frame']

root_dir = '/Users/celiaberon/GitHub/mouse_bandit/data/trial_data'

full_name = session_name + '_trials.csv'

path_name = os.path.join(root_dir,full_name)

trial_df = pd.read_csv(path_name,names=columns)

trial_df.head(5)

beliefs = hob.predictBeliefBySession(record_path, session_name, mouse_id)

columns.append('Belief')
trial_df['Belief'] = beliefs

trial_df.head(5)

plt.plot(trial_df['Right Reward Prob'])
plt.plot(trial_df['belief'])

feature_matrix = bp.create_feature_matrix(trial_df,10,mouse_id,session_name,feature_names='Default',imaging=True)

beliefs_feat_mat = hob.predictBeliefFeatureMat(feature_matrix, 10)

feature_matrix['Belief'] = beliefs_feat_mat

feature_matrix.head(5)

feature_matrix[['10_Port','10_ITI','10_trialDuration']].head(5)

def extract_frames(df, cond1_name, cond1=False, cond2_name=False, cond2=False, frame_type='decision_frame'):
    if type(cond2_name)==str:
        frames = (df[((df[cond1_name] == cond1) 
                    & (df[cond2_name] == cond2))][frame_type])
        return frames
    else:
        frames =(df[(df[cond1_name] == cond1)][frame_type])
        return frames

cond1_name = 'Switch'
cond1_a = 0
cond1_b = 1
cond2_name = 'Decision'
cond2_a = 0
cond2_b = 1

conditions_1 = [cond1_a, cond1_b]

extension = 30

frames_center_1a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_a, 'center_frame')
frames_decision_1a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_a, 'decision_frame')

frames_center_1b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_a, 'center_frame')
frames_decision_1b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_a, 'decision_frame')

start_stop_times_1a = [[frames_center_1a - extension], [frames_decision_1a + extension]] # start 10 frames before center poke
start_stop_times_1b = [[frames_center_1b - extension], [frames_decision_1b + extension]] # start 10 frames before center poke

frames_center_1b

frames_center_2a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_b, 'center_frame')
frames_decision_2a = extract_frames(feature_matrix, cond1_name, cond1_a, cond2_name, cond2_b, 'decision_frame')

frames_center_2b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_b, 'center_frame')
frames_decision_2b = extract_frames(feature_matrix, cond1_name, cond1_b, cond2_name, cond2_b, 'decision_frame')

start_stop_times_2a = [[frames_center_2a - extension], [frames_decision_2a + extension]] # start 10 frames before center poke
start_stop_times_2b = [[frames_center_2b - extension], [frames_decision_2b + extension]] # start 10 frames before center poke

#plt.plot(neuron.C_raw[0, preStart:trialDecision])
nNeurons = neuron.C.shape[0]
ca_data_path = '/Volumes/Neurobio/MICROSCOPE/Celia/data/k7_03142017_test/neuron_results.mat'
events = cc.detectEvents(ca_data_path)
neuron.C_raw = np.copy(events)

# remove neurons that have NaNs
nan_neurons = np.where(np.isnan(neuron.C_raw))[0]
nan_neurons = np.unique(nan_neurons)
good_neurons = [x for x in range(0, nNeurons) if x not in nan_neurons]

nNeurons = len(good_neurons) # redefine number of neurons
nTrials_1 = [len(start_stop_times_1a[0][0]), len(start_stop_times_1b[0][0])] # number of trials

# iterate through to determine duration between preStart and postDecision for each trial
window_length_1a = []
window_length_1b = []
for i in range(0,nTrials_1[0]):
    window_length_1a.append((start_stop_times_1a[1][0].iloc[i] - start_stop_times_1a[0][0].iloc[i]))
for i in range(0,nTrials_1[1]):
    window_length_1b.append((start_stop_times_1b[1][0].iloc[i] - start_stop_times_1b[0][0].iloc[i]))

# find longest window between preStart and postDecision and set as length for all trials
max_window_1 = int([max((max(window_length_1a), max(window_length_1b)))][0])

nTrials_2 = [len(start_stop_times_2a[0][0]), len(start_stop_times_2b[0][0])] # number of trials

# iterate through to determine duration between preStart and postDecision for each trial
window_length_2a = []
window_length_2b = []
for i in range(0,nTrials_2[0]):
    window_length_2a.append((start_stop_times_2a[1][0].iloc[i] - start_stop_times_2a[0][0].iloc[i]))
for i in range(0,nTrials_2[1]):
    window_length_2b.append((start_stop_times_2b[1][0].iloc[i] - start_stop_times_2b[0][0].iloc[i]))

# find longest window between preStart and postDecision and set as length for all trials
max_window_2 = int([max((max(window_length_2a), max(window_length_2b)))][0])

norm_trace = np.zeros((len(good_neurons),neuron.C_raw.shape[1]))
i = 0
for iNeuron in good_neurons:
    norm_trace[i,:] = (neuron.C_raw[iNeuron,:] - neuron.C_raw[iNeuron,:].min()) / (neuron.C_raw[iNeuron,:].max() - neuron.C_raw[iNeuron,:].min())
    i+=1

plt.hist(norm_trace)

#temp = [neuron.C_raw[x,:] - neuron.C_raw[x,:].min() for x in good_neurons]

neuron.C_raw = norm_trace

start_stop_times_1 = [start_stop_times_1a, start_stop_times_1b]
aligned_start_1 = np.zeros((np.max(nTrials_1), max_window_1, nNeurons, 2))
mean_center_poke_1 = np.zeros((max_window_1, nNeurons, 2))
norm_mean_center_1 = np.zeros((mean_center_poke_1.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    count = 0
    for iNeuron in range(len(good_neurons)):
        for iTrial in range(0,nTrials_1[i]):
            aligned_start_1[iTrial,:, count, i] = neuron.C_raw[iNeuron, int(start_stop_times_1[i][0][0].iloc[iTrial]):(int(start_stop_times_1[i][0][0].iloc[iTrial])+max_window_1)]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_center_poke_1[:,:,i]= np.mean(aligned_start_1[0:nTrials_1[i],:,:,i], axis=0)
   

start_stop_times_2 = [start_stop_times_2a, start_stop_times_2b]
aligned_start_2 = np.zeros((np.max(nTrials_2), max_window_2, nNeurons, 2))
mean_center_poke_2 = np.zeros((max_window_2, nNeurons, 2))
norm_mean_center_2 = np.zeros((mean_center_poke_2.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to center poke
    count = 0
    for iNeuron in range(len(good_neurons)):
        for iTrial in range(0,nTrials_2[i]):
            aligned_start_2[iTrial,:, count, i] = neuron.C_raw[iNeuron, int(start_stop_times_2[i][0][0].iloc[iTrial]):(int(start_stop_times_2[i][0][0].iloc[iTrial])+max_window_2)]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_center_poke_2[:,:,i]= np.mean(aligned_start_2[0:nTrials_2[i],:,:,i], axis=0)
    
   



agg_max = np.array([np.max(mean_center_poke_1,axis=0), np.max(mean_center_poke_2,axis=0)])
agg_max = np.max(np.max(agg_max, axis = 2), axis=0)

agg_min = np.array([np.min(mean_center_poke_1,axis=0), np.min(mean_center_poke_2,axis=0)])
agg_min = np.min(np.min(agg_min, axis = 2), axis=0)

for i in [0,1]:

    for iNeuron in range(0,nNeurons):
        norm_mean_center_1[:,iNeuron, i] = (mean_center_poke_1[:,iNeuron, i] - agg_min[iNeuron])/(agg_max[iNeuron] - agg_min[iNeuron])
        
    for iNeuron in range(0,nNeurons):
        norm_mean_center_2[:,iNeuron, i] = (mean_center_poke_2[:,iNeuron, i] - agg_min[iNeuron])/(agg_max[iNeuron] - agg_min[iNeuron])
    

for i in [0,1]:

    plt.figure(figsize=(8,8))
    plt.subplot(2,2,(2*i)+2-1)  
    plt.imshow(np.transpose(mean_center_poke_1[:,:,i])), plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    plt.title("%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_a, nTrials_1[i])) 
    plt.axis('tight')
    
    plt.subplot(2,2,(2*i)+2)  
    plt.imshow(np.transpose(mean_center_poke_2[:,:,i])), plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    plt.title("%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_b, nTrials_2[i])) 
    plt.axis('tight')
    

# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
for i in [0,1]:
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,(2*i)+2-1)  
    plt.imshow(np.transpose(norm_mean_center_1[:,:,i])), plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    title = "%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_a, nTrials_1[i])
    plt.title(title)
    plt.axis('tight')
    
    plt.subplot(2,2,(2*i)+2)  
    plt.imshow(np.transpose(norm_mean_center_2[:,:,i])), plt.colorbar()
    plt.axvline(x=extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frame (center poke at %s)' % extension)
    plt.ylabel('Neuron ID')
    title = "%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_b, nTrials_2[i])
    plt.title(title)
    plt.axis('tight')

sample_neuron = 10

plt.figure(figsize=(10,10))
plt.imshow(aligned_start_1[0:nTrials_1[0],:,sample_neuron, 0])
plt.axvline(x=extension, color='white', linestyle = '--', linewidth=.9)
plt.ylabel('Trial Number')
plt.xlabel('Frame (center poke at 10)')
plt.scatter((frames_decision_1a)-(frames_center_1a)+extension,range(0,nTrials_1[0]), color='white', marker = '|', s=10)
plt.title('%s = %s\n%s = %s\nNeuron ID = %s' % (cond1_name, conditions_1[0], cond2_name, cond2_a, sample_neuron))
plt.axis('tight')

aligned_decision_1 = np.zeros((np.max(nTrials_1), max_window_1, nNeurons, 2))
mean_decision_1 = np.zeros((max_window_1, nNeurons, 2))
norm_mean_decision_1 = np.zeros((mean_decision_1.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to decision poke
    count = 0
    for iNeuron in good_neurons:
        for iTrial in range(0,nTrials_1[i]):
            aligned_decision_1[iTrial,:, count, i] = neuron.C_raw[iNeuron, int(start_stop_times_1[i][1][0].iloc[iTrial])-max_window_1:(int(start_stop_times_1[i][1][0].iloc[iTrial]))]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_decision_1[:,:,i]= np.mean(aligned_decision_1[0:nTrials_1[i],:,:,i], axis=0)

aligned_decision_2 = np.zeros((np.max(nTrials_2), max_window_2, nNeurons, 2))
mean_decision_2 = np.zeros((max_window_2, nNeurons, 2))
norm_mean_decision_2 = np.zeros((mean_decision_2.shape[0], nNeurons, 2))

for i in [0,1]:

    # create array containing segment of raw trace for each neuron for each trial 
    # aligned to decision poke
    count = 0
    for iNeuron in good_neurons:
        for iTrial in range(0,nTrials_2[i]):
            aligned_decision_2[iTrial,:, count, i] = neuron.C_raw[iNeuron, int(start_stop_times_2[i][1][0].iloc[iTrial])-max_window_2:(int(start_stop_times_2[i][1][0].iloc[iTrial]))]
        count = count+1

    # take mean of fluorescent traces across all trials for each neuron, then normalize for each neuron
    mean_decision_2[:,:,i]= np.mean(aligned_decision_2[0:nTrials_2[i],:,:,i], axis=0)

for i in [0,1]:
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,(2*i)+2-1)  
    plt.imshow(np.transpose(mean_decision_1[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window_1-extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frames (decision poke at %s)' % (max_window_1-extension))
    plt.ylabel('Neuron ID')
    plt.title("%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_a, nTrials_1[i]))
    plt.axis('tight')
    
    plt.subplot(2,2,(2*i)+2)  
    plt.imshow(np.transpose(mean_decision_2[:,:,i])), plt.colorbar()
    plt.xlabel('Frames (decision poke at %s)' % (max_window_2-extension))
    plt.axvline(x=max_window_2-extension, color='k', linestyle = '--', linewidth=.9)
    plt.ylabel('Neuron ID')
    plt.title("%s = %s\n %s = %s\nNum trials = %.0f" % (cond1_name, conditions_1[i], cond2_name, cond2_b, nTrials_2[i]))
    plt.axis('tight')

# heatmap for all neurons (each neuron represented by avg fluorescence across all trials)
for i in [0,1]:
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,(2*i)+2-1)  
    plt.imshow(np.transpose(norm_mean_decision_1[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window_1-extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frames (decision poke at %s)' % (max_window_1-extension))
    plt.ylabel('Neuron ID')
    title = "%s = %s\n %s = %s" % (cond1_name, conditions_1[i], cond2_name, cond2_a)
    plt.title(title + '\nNum trials: %.0f' % nTrials_1[i])
    plt.axis('tight')
    
    plt.subplot(2,2,(2*i)+2)  
    plt.imshow(np.transpose(norm_mean_decision_2[:,:,i])), plt.colorbar()
    plt.axvline(x=max_window_2-extension, color='k', linestyle = '--', linewidth=.9)
    plt.xlabel('Frames (decision poke at %s)' % (max_window_2-extension))
    plt.ylabel('Neuron ID')
    title = "%s = %s\n %s = %s" % (cond1_name, conditions_1[i], cond2_name, cond2_b)
    plt.title(title + '\nNum trials: %.0f' % nTrials_2[i])
    plt.axis('tight')

plt.plot(norm_mean_decision_1[:,10,1])
norm_mean_decision_1.shape

# plot the difference between two conditions for aligned to center poke
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(np.transpose(norm_mean_center_1[:,:,0] - norm_mean_center_1[:,:,1])), plt.colorbar()
plt.axvline(x=extension, color='white', linestyle = '--', linewidth=.9)
plt.ylabel('Neuron ID')
plt.xlabel('Frames (center poke at %s)' % extension)
plt.title('%s(%s) - %s(%s)\n %s = %s' % (cond1_name, conditions_1[0], cond1_name, conditions_1[1], cond2_name, cond2_a))

plt.subplot(1,2,2)
plt.imshow(np.transpose(norm_mean_center_2[:,:,0] - norm_mean_center_2[:,:,1])), plt.colorbar()
plt.axvline(x=extension, color='white', linestyle = '--', linewidth=.9)
plt.ylabel('Neuron ID')
plt.xlabel('Frames (center poke at %s)' % extension)
plt.title('%s(%s) - %s(%s)\n %s = %s' % (cond1_name, conditions_1[0], cond1_name, conditions_1[1], cond2_name, cond2_b))


# plot the difference between two conditions for aligned to decision poke
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(np.transpose(norm_mean_decision_1[:,:,0] - norm_mean_decision_1[:,:,1])), plt.colorbar()
plt.axvline(x=max_window_1-extension, color='white', linestyle = '--', linewidth=.9)
plt.xlabel('Frames (decision poke at %s)' % (max_window_1-extension))
plt.ylabel('Neuron ID')
plt.title('%s(%s) - %s(%s)\n %s = %s' % (cond1_name, conditions_1[0], cond1_name, conditions_1[1], cond2_name, cond2_a))

plt.subplot(1,2,2)
plt.imshow(np.transpose(norm_mean_decision_2[:,:,0] - norm_mean_decision_2[:,:,1])), plt.colorbar()
plt.axvline(x=max_window_2-extension, color='white', linestyle = '--', linewidth=.9)
plt.xlabel('Frames (decision poke at %s)' % (max_window_2-extension))
plt.ylabel('Neuron ID')
plt.title('%s(%s) - %s(%s)\n %s = %s' % (cond1_name, conditions_1[0], cond1_name, conditions_1[1], cond2_name, cond2_b))



help(detectEvents)

