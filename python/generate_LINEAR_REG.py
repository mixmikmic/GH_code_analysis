# Makes possible to show the output from matplotlib inline
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt

# Makes the figures in the PNG format:
# For more information see %config InlineBackend
get_ipython().magic("config InlineBackend.figure_formats=set([u'png'])")

plt.rcParams['figure.figsize'] = 5, 10

import numpy
import sys
import os

from sklearn import linear_model

import save_load_file as slf

import membrane_lowpass_md
membrane_lowpass = membrane_lowpass_md.membrane_lowpass

import time

# Loads the modules and starts the object to be used with the parallel processing iPython stuff...

# Remember to start the clusters:
# https://ipyparallel.readthedocs.org/en/latest/process.html

#from ipyparallel import Client

#cli = Client()

#lbview = cli.load_balanced_view()
# dview = cli[:]

#
# Controls if the results are saved to a file
#

save2file = True

#
# This is the low-pass filter (neuron membrane) applied to the outputs:
membrane_time_constant = 30E-3

# Reducing this value reduces the "memory" of the membrane.

NofN = 8*5*5 # total number of neurons in the output
sim_step_time = 2E-3 #simulation step time (in seconds)

#simulation type
platform = "realBaxter"
trial_number = 0
total_trials = 100

#
# Controls if the results are saved to a file
#
save2file = True

#simulation type
#platform = "gazeboBaxter"
platform = "realBaxter"

##--data filenames--
input_filename = "XY_movement_square_+0.5cm"
#input_filename = "XY_movement_square_HIGHTABLE3_free"

input_folder = "PIDs_DATA_FOR_THE_LSM"
data_folder = "PIDs_LSM"
sim_type = "with_PID_0.5cm"

total_trials_available = 100

number_of_trajectories = 1

disconnected = False

size_distr=10000

trial_array = numpy.arange(0,total_trials_available,dtype=numpy.int)
# numpy.random.shuffle(trial_array)

#
# Creates a function to read the spikes in a (multiple process) parallel way.
#
#@lbview.parallel(block=True)
def reads_files(filename):
    import save_load_file as slf 
    return slf.load_from_file_gz(filename)

#
# Here the liquid index is defined
#
# REMEMBER: it goes from ZERO to (number_of_liquids-1)

lsm_i = 4

get_ipython().run_cell_magic('time', '', '# \n# Loads the spikes generated by the liquid with index=lsm_i\n# \n\nfilenames = []\noutput_spikes_simulation = []\nfor pos_i in xrange(number_of_trajectories):\n    for run_i in trial_array[:total_trials]:\n        filenames.append(os.getcwd()+"/"+data_folder+"/"+"sensors_LSM_"+str(lsm_i)+"_"+str(run_i)+".gzpickle")\n        # The format of simulated_values is (a list of tuples):\n        # [\n        # current time (in ms),    =>index 1\n        # numpy.array with spikes, =>index 2\n        # ]\n    output_spikes_simulation.append(map(reads_files,filenames))')

# 
# Prints some values (first and last time steps), just to make sure it worked...
#

print output_spikes_simulation[0][0][0]

print output_spikes_simulation[0][-1][-1]

total_trajectories,total_trials,total_steps,sim_step_time = len(output_spikes_simulation),len(output_spikes_simulation[0]),len(output_spikes_simulation[0][0]), (output_spikes_simulation[0][0][1][0]-output_spikes_simulation[0][0][0][0])
total_trajectories,total_trials,total_steps,sim_step_time

get_ipython().run_cell_magic('time', '', '# Generates the FILTERED (membrane low-pass filter) data to be used with the linear regression\n\n# The first index of the matrix is the trajectory\n# Example (each trajectory has 250 steps): \n# linalg_matrix_filtered[0][0:250] => is the first experiment of the first trajectory\n# linalg_matrix_filtered[0][250:250*2] => is the second experiment of the first trajectory\n\navoid_n = 0 # Avoids the avoid_n steps after step=0 (always ignored)\n            # This type of LSM receives during the first N steps the correct values before it\n            # starts generating the rest of the time series.\n            # N = avoid_n+1\n\n# t_idx = 0 # 0=>first trajectory\n# e_idx = 0 # 0=>first experiment\nlinalg_matrix_filtered = numpy.zeros((total_trajectories, total_trials*(total_steps-avoid_n), NofN),dtype=numpy.float)\n\nfor t_idx in range(total_trajectories): # goes through all the trajectories\n    for e_idx in range(total_trials): # goes through all the trials\n        m_v=membrane_lowpass(NofN,membrane_time_constant) # Initialize the membrane for each new trial\n        for i in range(1,total_steps): # ignores the first output from the network (noisy, uncorrelated with input)\n            if (output_spikes_simulation[t_idx][e_idx][i][1]).size>0:\n                m_v.process_spikes(output_spikes_simulation[t_idx][e_idx][i][1],\\\n                                   output_spikes_simulation[t_idx][e_idx][i][0])\n            if i >= avoid_n: #useless as i starts at 1 and avoid_n=1...\n                linalg_matrix_filtered[t_idx][(i-avoid_n)+\\\n                                              (total_steps-avoid_n)*e_idx]=m_v.check_values(i*sim_step_time) # Saves the membrane state at each time step\n\n\n# This is another good candidate to be vectorized...')

print linalg_matrix_filtered.shape

pid_hist=numpy.ndarray((total_trials,1000))

plt.figure(figsize=(10,5))
for trial_number in range(total_trials):
    #reads stuff
    pidF_hist=numpy.copy(numpy.load(input_folder+"/"+sim_type+"PID_Force_hist_trial"+str(trial_number)+".npy"))
    pidR_hist=numpy.copy(numpy.load(input_folder+"/"+sim_type+"PID_Range_hist_trial"+str(trial_number)+".npy"))
    pid_hist[trial_number] =  pidF_hist+pidR_hist
    plt.plot(pid_hist[trial_number])
plt.title("Closed loop gain from both PID controllers")
plt.ylabel("gain")
plt.xlabel("step")
plt.show()

pid_hist.shape

y_sensor = []
for ti in range(total_trials):
        y_sensor=numpy.concatenate((y_sensor,pid_hist[ti]))
print y_sensor.shape

X_matrix=linalg_matrix_filtered    

X_matrix.shape

# Prepare the matrix to be used with numpy linear regression:
X_reshaped=X_matrix.reshape(y_sensor.shape[0],NofN)

# Creates an empty matrix with an extra collumn with ones (numpy.linalg.lstsq demands this...)
X_reshaped=numpy.ones((X_reshaped.shape[0],X_reshaped.shape[1]+1))

# Writes the values in to the first NofN collumns
X_reshaped[:,:NofN]=X_matrix.reshape(y_sensor.shape[0],NofN)

# Now the reshaped matrix has an extra collumn:
X_reshaped.shape

X=X_reshaped

numpy.shape(X)[0]*numpy.shape(X)[1]

X.max(),X.min(),y_sensor.max(),y_sensor.min()

X.shape, y_sensor.shape

X[:,-1] # Shows the last column made of ones...

filename = "./"+data_folder+"/"+"sensors_LSM_"+str(lsm_i)+".gzpickle"
print filename


get_ipython().run_cell_magic('time', '', '\n# Non parallel linear regression (trying to solve the pickles problem when I have too many trials (more than 300...))\nfrom sklearn import linear_model\n\nX=X[:,:NofN] # Cuts out the extra ones used only for \n\nsklg = linear_model.Ridge()\ny=y_sensor.T\n\nsklg.fit(X,y)')

sklg.intercept_

if save2file:
    slf.save_to_file([sklg.coef_,sklg.intercept_],filename)
[c_sensors,r_sensors] = slf.load_from_file(filename)

s_calculated=X_matrix.reshape(y_sensor.shape[0],NofN).dot(c_sensors)+r_sensors

X_matrix.shape

# This dictionary is used to automate the figures generation
joints_dict={'PID':(y_sensor,s_calculated) }

# Plots the inputs and the outputs side-by-side

lsm_out = 'PID'

y_1 = joints_dict[lsm_out][0]
y1_calculated = joints_dict[lsm_out][1]

offset11 = y_1.shape[0]/number_of_trajectories
offset12 = y_1.shape[0]/number_of_trajectories/total_trials
print offset11,offset12

fig=plt.figure(figsize =(20,10));

for trajectory in xrange(number_of_trajectories):
# trajectory=0 # goes from 0 to 3

    ymax=numpy.array([y_1[trajectory*(total_trials*total_steps):total_steps*(1+trajectory*(total_trials*total_steps))].max()]).max()
    ymin=numpy.array([y_1[trajectory*(total_trials*total_steps):total_steps*(1+trajectory*(total_trials*total_steps))].min()]).min()

    ymaxLSM=numpy.array([y1_calculated[trajectory*(total_trials*total_steps):total_steps*(1+trajectory*(total_trials*total_steps))].max()]).max()
    yminLSM=numpy.array([y1_calculated[trajectory*(total_trials*total_steps):total_steps*(1+trajectory*(total_trials*total_steps))].min()]).min()


    plt.subplot(number_of_trajectories,2,2*trajectory+1)    
    plt.plot(y_1[trajectory*(offset11):offset11*trajectory+offset12],'b')

    plt.ylim(ymin-abs(ymin)/5.,ymax+abs(ymax)/5.)
    plt.title("Original analog joints "+ str(lsm_out) +" - trajectory " + str(trajectory+1))

    plt.subplot(number_of_trajectories,2,2*trajectory+2)
    plt.plot(y1_calculated.reshape(number_of_trajectories,total_trials,total_steps-avoid_n)[trajectory,:,:].mean(axis=0)+y1_calculated.reshape(number_of_trajectories,total_trials,total_steps-avoid_n)[trajectory,:,:].std(axis=0),'r')
    plt.plot(y1_calculated.reshape(number_of_trajectories,total_trials,total_steps-avoid_n)[trajectory,:,:].mean(axis=0)-y1_calculated.reshape(number_of_trajectories,total_trials,total_steps-avoid_n)[trajectory,:,:].std(axis=0),'r')    
    plt.plot(y1_calculated.reshape(number_of_trajectories,total_trials,total_steps-avoid_n)[trajectory,:,:].mean(axis=0),'b')    
    plt.ylim(ymin-abs(ymin)/5.,ymax+abs(ymax)/5.)
    plt.title("LSM output (trials mean/std values) joints "+ str(lsm_out) +" - trajectory " + str(trajectory+1))

# fig.subplots_adjust(bottom=0,hspace=.6) # Adjust the distance between subplots
plt.tight_layout()
# plt.savefig("./"+base_dir+"/"+sim_set+"/readout_test_s0_s1_"+sim_set+".pdf")
plt.show()

plt.figure(figsize=(15,5))
plt.plot(c_sensors,'v')
plt.title("Coefficients")
plt.show()

# Independent terms
r_sensors
# Independent terms close to zero are the result of the bias extraction (I force the curves to start at zero)



