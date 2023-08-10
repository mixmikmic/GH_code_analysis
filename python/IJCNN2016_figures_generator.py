import numpy
import os
import save_load_file as slf

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib 

shape_name = ["circle","square","triangle"]

sim_set = shape_name[0] # basically is the name of the folder where the data is read/saved
base_dir = "BaxterArm_VREP_simulation_data"

XY_movement = []
for i in range(len(shape_name)):
    XY_movement.append(slf.load_from_file(base_dir+"/"+shape_name[i]+"/"+shape_name[i]+".pickle"))

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# font = {'weight' : 'normal',
#         'size'   : 20}

# matplotlib.rc('font', **font)

# matplotlib.rcdefaults() # restores to the default values

# http://matplotlib.org/api/matplotlib_configuration_api.html#matplotlib.rc

font = {'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

plt.figure(figsize =(10,10))
for i in range(len(shape_name)):
    plt.plot(XY_movement[i][::5,0],XY_movement[i][::5,1],'.',markersize=7)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
# plt.subplots_adjust(left=0, bottom=.1, right=1, top=1,wspace=.2, hspace=.5)
# plt.savefig(os.getcwd()+"/"+"initial_shapes.pdf", bbox_inches='tight',pad_inches=.1)
plt.show()

len(XY_movement[2][::5,0])

# 
# Defines which shape is going to be used to generate the figures
# 0=>circle; 1=>square; 2=>triangle
#
shape_i=2

joint_positions=numpy.load(base_dir+"/"+shape_name[shape_i]+"/XY_movement_"+shape_name[shape_i]+".npy")


font = {'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

#
# Plots the original curves
#
fig=plt.figure(figsize =(15,20))

# plt.subplot(7,1,1)
# plt.plot(XY_movement[shape_i][:,0],XY_movement[shape_i][:,1])
# plt.title("X and Y")

plt.subplot(7,1,2)
plt.plot(XY_movement[shape_i][:,0])
plt.xlim(0,len(XY_movement[shape_i]))
plt.title("X (m)")

plt.subplot(7,1,3)
plt.plot(XY_movement[shape_i][:,1])
plt.xlim(0,len(XY_movement[shape_i]))
plt.title("Y (m)")

plt.subplot(7,1,4)
plt.plot(joint_positions[:,0])
plt.xlim(0,len(joint_positions))
plt.title("Joint S0 (rad)")

plt.subplot(7,1,5)
plt.plot(joint_positions[:,1])
plt.xlim(0,len(joint_positions))
plt.title("Joint S1 (rad)")

plt.subplot(7,1,6)
plt.plot(joint_positions[:,2])
plt.xlim(0,len(joint_positions))
plt.title("Joint E1 (rad)")

plt.subplot(7,1,7)
plt.plot(joint_positions[:,3])
plt.xlim(0,len(joint_positions))
plt.title("Joint W1 (rad)")

plt.xlabel("simulation step")

plt.subplots_adjust(left=0, bottom=.1, right=1, top=1,wspace=.2, hspace=.5)
# plt.savefig(os.getcwd()+"/"+"example_joints.pdf", bbox_inches='tight',pad_inches=.1)
plt.show()

#
# The variables set here will be used all along the notebook
#

simulation_type = "parallel"
# simulation_type = "serial"
# simulation_type = "true_serial"

lsm_i = "ALL"
# lsm_i = 0
trial_number = 0

filename = base_dir+"/"+shape_name[shape_i]+"/joint_angles_mean_"+simulation_type+"_"+str(lsm_i)+"_"+str(trial_number)+".npy"
joint_angles_mean = numpy.load(filename)

filename = base_dir+"/"+shape_name[shape_i]+"/joint_angles_individual"+simulation_type+"_"+str(lsm_i)+"_"+str(trial_number)+".npy"
joint_angles_individual = numpy.load(filename)

filename = base_dir+"/"+shape_name[shape_i]+"/baxter_xyz_joint_angles_mean_"+simulation_type+"_"+str(lsm_i)+"_"+str(trial_number)+".npy"
xyz = numpy.load(filename)

# simulation_results = numpy.array(joint_angles_mean)
# joint_names = ["s0","s1","e1","w1"]

# font = {'weight' : 'normal',
#         'size'   : 15}

# matplotlib.rc('font', **font)

# plt.figure(figsize =(15,20))
# for ji in range(4):
#     plt.subplot(411+ji)
#     plt.plot(joint_positions[:,ji],'b--',linewidth=4,label="Original")
#     plt.plot(simulation_results[:,ji],'g-',linewidth=2,label="Calculated")
#     plt.title("Joint "+joint_names[ji] + " - trial_number:"+str(trial_number))
#     plt.legend()
    
# plt.xlabel("simulation step")
# plt.subplots_adjust(left=0, bottom=.1, right=1, top=1,wspace=.2, hspace=.5)
# plt.savefig(os.getcwd()+"/"+"comparing_joints.pdf", bbox_inches='tight',pad_inches=.1)

# plt.show()

# font = {'weight' : 'normal',
#         'size'   : 15}

# matplotlib.rc('font', **font)

# plt.figure(figsize =(15,10))
# plt.subplot(2,1,1)
# plt.plot(XY_movement[shape_i][:,0]-XY_movement[shape_i][0,0],'b--',linewidth=4,label="Original")
# plt.plot(xyz[:,0]-xyz[0,0],'g-',linewidth=2,label="Generated")
# # plt.xlim(0,len(XY_movement[shape_i]))
# plt.title("X (m)" + " - trial_number:"+str(trial_number))
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(XY_movement[shape_i][:,1]-XY_movement[shape_i][0,1],'b--',linewidth=4,label="Original")
# plt.plot(xyz[:,1]-xyz[0,1],'g-',linewidth=2,label="Generated")
# # plt.xlim(0,len(XY_movement[shape_i]))
# plt.title("Y (m)" + " - trial_number:"+str(trial_number))
# plt.legend()

# plt.xlabel("simulation step")
# plt.subplots_adjust(left=0, bottom=.1, right=1, top=1,wspace=.2, hspace=.2)
# plt.savefig(os.getcwd()+"/"+"comparing_xy.pdf", bbox_inches='tight',pad_inches=.1)
# plt.show()

# font = {'weight' : 'normal',
#         'size'   : 15}

# matplotlib.rc('font', **font)

# plt.figure(figsize =(15,15))
# plt.plot(XY_movement[shape_i][:,0],XY_movement[shape_i][:,1])
# plt.plot(xyz[:,0]-xyz[0,0]+XY_movement[shape_i][0,0],xyz[:,1]-xyz[0,1]+XY_movement[shape_i][0,1],'.-')
# plt.title("Final Shape")
# plt.show()

xyz_pos = []
for trial_number in range(10):
    filename = base_dir+"/"+shape_name[shape_i]+"/baxter_xyz_joint_angles_mean_"+simulation_type+"_"+str(lsm_i)+"_"+str(trial_number)+".npy"
    xyz_pos.append(numpy.load(filename))

xyz_pos = numpy.array(xyz_pos)

# Translates to match the original initial value (as the simulations always start at that value)
xyz_pos[:,:,0]+=-xyz[0,0]+XY_movement[shape_i][0,0]
xyz_pos[:,:,1]+=-xyz[0,1]+XY_movement[shape_i][0,1]

# xyz_pos_mean = xyz_pos.mean(axis=0)
# xyz_pos_std = xyz_pos.std(axis=0)
# xyz_pos_stderr = xyz_pos_std/numpy.sqrt(xyz_pos.shape[0])

# font = {'weight' : 'normal',
#         'size'   : 15}

# matplotlib.rc('font', **font)

# plt.figure(figsize =(10,10))
# plt.plot(xyz_pos_mean[:,0],xyz_pos_mean[:,1],'g')
# plt.plot(xyz_pos_mean[:,0]-xyz_pos_stderr[:,0],xyz_pos_mean[:,1]-xyz_pos_stderr[:,1],'b')
# plt.plot(xyz_pos_mean[:,0]+xyz_pos_stderr[:,0],xyz_pos_mean[:,1]+xyz_pos_stderr[:,1],'r')
# plt.plot(XY_movement[shape_i][:,0],XY_movement[shape_i][:,1],'k--',linewidth=2)

# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Cartesian Movement Generated by Baxter (mean / stderr) - "+simulation_type)
# # plt.savefig(os.getcwd()+"/"+"comparing_joints.pdf", bbox_inches='tight',pad_inches=.1)
# plt.show()

font = {'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)

plt.figure(figsize =(10,10))
for i in range(10):
    plt.plot(xyz_pos[i,:,0],xyz_pos[i,:,1])
plt.plot(XY_movement[shape_i][:,0],XY_movement[shape_i][:,1],'k--',linewidth=2)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
# plt.title("Cartesian Movement Generated by Baxter (all trials) - "+simulation_type)
# plt.savefig(os.getcwd()+"/"+"final_xy_"+shape_name[shape_i][:-1]+"_"+simulation_type+".pdf", bbox_inches='tight',pad_inches=.1)
plt.show()

import sys
import os
import dtw_C

trials_cost = []
for trial_n in range(10):
    original_drawing = numpy.copy(XY_movement[shape_i])
    testing_drawing = numpy.copy(xyz_pos[trial_n,:,0:2])

    dist_matrix = dtw_C.distances_matrix(original_drawing,testing_drawing)

    accumulated_cost = dtw_C.accumulated_cost_matrix(original_drawing,testing_drawing,dist_matrix)

    _, cost = dtw_C.path_cost_calculation(original_drawing,testing_drawing, dist_matrix, accumulated_cost)
    trials_cost.append(cost)
#     trials_cost.append(accumulated_cost[-1,-1]/sum(accumulated_cost.shape))

filename = base_dir+"/"+shape_name[shape_i]+"/baxter_xyz_dtw_"+simulation_type+".pickle"
# slf.save_to_file([numpy.copy(xyz_pos),numpy.array(trials_cost)],filename)

filename = base_dir+"/"+shape_name[shape_i]+"/baxter_xyz_dtw_parallel.pickle"
xyz_pos_parallel,trials_cost_parallel = slf.load_from_file(filename)

filename = base_dir+"/"+shape_name[shape_i]+"/baxter_xyz_dtw_true_serial.pickle"
xyz_pos_serial,trials_cost_serial = slf.load_from_file(filename)

max_value = (numpy.array([trials_cost_serial,trials_cost_parallel])).max()

plt.figure(figsize =(20,10))

plt.subplot2grid((2,10),(0, 0))
plt.plot(xyz_pos_parallel[0,:,0],xyz_pos_parallel[0,:,1],'b:')
plt.plot(xyz_pos_serial[0,:,0],xyz_pos_serial[0,:,1],'r--')
#     plt.axis('off')
plt.xlim(-0.15,0.15)
plt.ylim(-0.15,0.15)
plt.xticks([0],[1])
plt.yticks([],[])# avoids white spaces

for i in range(1,10):
    plt.subplot2grid((2,10),(0, i))
    plt.plot(xyz_pos_parallel[i,:,0],xyz_pos_parallel[i,:,1],'b:')
    plt.plot(xyz_pos_serial[i,:,0],xyz_pos_serial[i,:,1],'r--')
#     plt.axis('off')
    plt.xlim(-0.15,0.15)
    plt.ylim(-0.15,0.15)
    plt.xticks([0],[i+1])
    plt.yticks([],[])# avoids white spaces

plt.subplot2grid((2,10),(1, 0),colspan=10)

plt.plot(range(1,11),trials_cost_parallel/max_value,'bo-', markersize=10)
plt.plot(range(0,12),[numpy.mean(trials_cost_parallel/max_value)]*12,'b:')
print "Average - Parallel:",numpy.mean(trials_cost_parallel/max_value)

plt.plot(range(1,11),trials_cost_serial/max_value,'rs-', markersize=10)
plt.plot(range(0,12),[numpy.mean(trials_cost_serial/max_value)]*12,'r--')
print "Average - Serial:",numpy.mean(trials_cost_serial/max_value)

plt.xlim(0.5,10.5)
plt.ylabel('normalised cost')
plt.xlabel('trial number')
plt.xticks([0.5]+range(1,11),['']+range(1,11))
plt.ylim(-0.05,1.05)

# plt.savefig(os.getcwd()+"/"+"final_xy_"+shape_name[shape_i][:-1]+"_dtw.pdf", bbox_inches='tight',pad_inches=.1)
plt.show()

font = {'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)

plt.figure(figsize =(10,10))
for i in range(10):
    plt.plot(xyz_pos[i,:,2]*1000)

plt.plot([xyz_pos[0,0,2]*1000]*len(xyz_pos[0,:,2]),'k--',linewidth=2)

plt.plot([xyz_pos[:,:,2].max()*1000]*len(xyz_pos[0,:,2]),'k:',linewidth=2)
plt.plot([xyz_pos[:,:,2].min()*1000]*len(xyz_pos[0,:,2]),'k:',linewidth=2)

# plt.annotate(str(xyz_pos[0,0,2]*1000),(500,xyz_pos[0,0,2]*1000))
         
plt.xlabel("simulation step")
plt.ylabel("z (mm)")
# plt.title("Cartesian Movement Generated by Baxter (all trials) - "+simulation_type)
# plt.savefig(os.getcwd()+"/"+"final_z_"+shape_name[shape_i][:-1]+"_"+simulation_type+".pdf", bbox_inches='tight',pad_inches=.1)
plt.show()
print "z:",numpy.round(xyz_pos[0,0,2]*1000,2)
print "zmax:",numpy.round(xyz_pos[:,:,2].max()*1000,2),numpy.round(abs(xyz_pos[:,:,2].max()-xyz_pos[0,0,2])*1000,2)
print "zmin:",numpy.round(xyz_pos[:,:,2].min()*1000,2),numpy.round(abs(xyz_pos[:,:,2].min()-xyz_pos[0,0,2])*1000,2)

font = {'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)

plt.figure(figsize =(10,10))
plt.subplot(2,1,1)
i_range = range(10)
numpy.random.shuffle(i_range) #to solve the problem when the same colour appear twice and make it harder to analyse
for i in i_range:
    plt.plot(xyz_pos[i,:,0],linewidth=2,label="Generated")
plt.plot(XY_movement[shape_i][:,0],'k--',linewidth=5,label="Original")
# plt.xlim(0,len(XY_movement[shape_i]))
plt.title("X (m)")

plt.subplot(2,1,2)
for i in i_range:
    plt.plot(xyz_pos[i,:,1],linewidth=2,label="Generated")
plt.plot(XY_movement[shape_i][:,1],'k--',linewidth=5,label="Original")
# plt.xlim(0,len(XY_movement[shape_i]))
plt.title("Y (m)")

plt.xlabel("simulation step")

plt.subplots_adjust(left=0, bottom=.1, right=1, top=1,wspace=.2, hspace=.2)
# plt.savefig(os.getcwd()+"/"+"final_xy_time_"+shape_name[shape_i][:-1]+"_"+simulation_type+".pdf", bbox_inches='tight',pad_inches=.1)
plt.show()

