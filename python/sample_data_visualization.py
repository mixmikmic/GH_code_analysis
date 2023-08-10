import numpy as np
import pandas as pd
from ast import literal_eval

in_file = 'sample_data.csv'
sample = pd.read_csv(in_file, sep=';')
sample.ptsx = sample.ptsx.apply(literal_eval)
sample.ptsy = sample.ptsy.apply(literal_eval)

sample.head()

sample.describe()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
speed = pd.Series(sample.speed)
fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
p0 = ax1.plot(speed.index, speed, 'g', label='P controller')
p1 = ax1.plot(speed.index, np.ones(speed.count())*30.0, 'r', label='target')
ax1.set_xlabel("Time Steps", fontsize=10)
ax1.set_ylabel("MPH", fontsize=10)
plt.title('Speed PID Controller')
plt.legend((p0[0], p1[0]), ('controller', 'target'))
plt.show()

steering = pd.Series(sample.steering_angle)
steering = steering.round(1)

steerInfo = {}
for i in range(len(steering)):
    # get the current steering angle bucket
    if steering[i] == 0.:
        steer = '0.0'
    else:
        steer = str(steering[i])
    # try to see if there is a hash hit
    steerInstance = steerInfo.get(steer, {'count':0, 'samples':[]})
    # add to count
    count = steerInstance['count'] + 1
    # add to samples
    samples = steerInstance['samples']
    samples.append(i)
    # put in the last index
    steerInfo[steer] = {'lastIdx':i, 'count': count, 'steering_angle':steering[i], 'samples':samples}
    
# get the list of steer and sort them
sortedSteering = list(steerInfo.keys())
def compare_steer(steer):
    return steerInfo[steer]['steering_angle']
sortedSteering.sort(key=compare_steer)
steerCounts = [steerInfo[n]['count'] for n in sortedSteering]

n_steers = len(sortedSteering)

ind = np.arange(n_steers)
width = 0.8

fg, ax = plt.subplots(figsize=(n_steers, n_steers/2))
rects1 = ax.bar(ind+1.6, steerCounts, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel("Counts", fontsize=10)
ax.set_title("Steering Angles Distribution", fontsize=15)
ax.set_xticks(ind + width + 1.2)
ax.set_xticklabels(sortedSteering, fontsize=12)
ax.set_xlabel("Steering Angles", fontsize=10)
plt.show()

from tqdm import tqdm
import random

n_samples = 4

# size of each sample
fig, ax0 = plt.subplots(figsize=(n_samples*2.4, n_steers))
ax0.set_title("Steering Angles with 4 Random Image Samples", fontsize=15)
ax0.set_xticks([])
ax0.set_yticks([])
w_ratios = [1 for n in range(n_samples)]
w_ratios[:0] = [int(n_samples*0.4)]
h_ratios = [1 for n in range(n_steers)]

# gridspec
grid = gridspec.GridSpec(n_steers, n_samples+1, wspace=0.0, hspace=0.0, width_ratios=w_ratios, height_ratios=h_ratios)
steerset_pbar = tqdm(range(n_steers), desc='Steering Angle Image Samples', unit='angles')
for a in steerset_pbar:
    steer = str(sortedSteering[a])
    count = steerInfo[steer]['count']
    for b in range(n_samples+1):
        i = a*(n_samples+1) + b
        ax = plt.Subplot(fig, grid[i])
        if b == 0:
            ax.annotate('steering angle %s\nsample count: %d'%(steer, count), xy=(0,0), xytext=(0.0,0.5))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        elif count < n_samples:
            if (b-1) < count:
                image=mpimg.imread(sample.image[b-1])
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)
        else:
            random_i = random.choice(steerInfo[steer]['samples'])
            image=mpimg.imread(sample.image[random_i])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    # hide the borders\
    if a == (n_steers-1):
        all_axes = fig.get_axes()
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
                
for sp in ax0.spines.values():
    sp.set_visible(False)

plt.show()

waypoint_file = "lake_track_waypoints.csv"
ml_waypoint_file = "lake_track_ml_generated_waypoints.csv"

with open(waypoint_file) as f:
    x_waypoint = []
    y_waypoint = []
    count = 0
    for line in f:
        if count > 0:
            data = line.split(',')
            x_waypoint.append(data[0])
            y_waypoint.append(data[1])
        count += 1
    
with open(ml_waypoint_file) as f:
    x_ml_waypoint = []
    y_ml_waypoint = []
    count = 0
    for line in f:
        if count > 0:
            data = line.split(',')
            x_ml_waypoint.append(data[0])
            y_ml_waypoint.append(data[1])   
        count += 1
        
x_start = [ x_ml_waypoint[0] ]
y_start = [ y_ml_waypoint[1] ]

plt.rcParams["figure.figsize"] = [16, 16]
img=mpimg.imread('track1_top_view.png')
p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p2 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p3 = plt.plot(x_start, y_start, 'ro', ms=5.0)
plt.title('Test Track Map with Waypoints')
plt.xlabel("X", fontsize=10)
plt.ylabel("Y", fontsize=10)
plt.legend((p2[0], p3[0]), ('Provided Waypoints', 'Starting Position'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_ml_waypoint, y_ml_waypoint, 'b', lw=0.5)
p2 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p3 = plt.plot(x_start, y_start, 'ro', ms=5.0)
plt.title('Test Track Map with Way Points and Ground Truth Path')
plt.xlabel("X", fontsize=10)
plt.ylabel("Y", fontsize=10)
plt.legend((p1[0], p2[0], p3[0]), ('gen0 path', 'Provided Waypoints', 'Starting Position'))
plt.show()

data = pd.read_csv('../gen1/training-progress.csv')
generation = 'g=1'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=40)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('Generation {} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (sessions=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = result_success
all_testing_sessions = testing_sessions

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

data = pd.read_csv('../gen2/training-progress.csv')
generation = 'g=2'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=40)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('Generation {} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = pd.concat([all_result_success, result_success])
all_testing_sessions += testing_sessions

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

data = pd.read_csv('../gen3/training-progress.csv')
generation = 'g=3'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=40)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('Generation {} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = pd.concat([all_result_success, result_success])
all_testing_sessions += testing_sessions

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

data = pd.read_csv('../gen4/training-progress.csv')
generation = 'g=4'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('Generation {} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = pd.concat([all_result_success, result_success])
all_testing_sessions += testing_sessions

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

data = pd.read_csv('../gen5/training-progress.csv')
generation = 'g=5'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('Generation {} Test Session Results'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('Generation {} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = pd.concat([all_result_success, result_success])
all_testing_sessions += testing_sessions

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('All Session Success Traces on Test Track Map')
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in all_result_success['session']:
    testing_sessions_end.append(all_result_success[all_result_success['session'] == j])
    if i < len(all_testing_sessions):
        plt.plot(all_testing_sessions[i]['x'], all_testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(i+1))
    i += 1
plt.legend(sessions, labels)
plt.show()

data = pd.read_csv('../speed/training-progress.csv')
generation = 'Speed training'

# create subsets
training_data = data[data['testing'] == False]
testing_data = data[data['testing'] == True]
testing_sessions = []
testing_sessions_end = []
for i in range(testing_data['session'].max()):
    testing_sessions.append(testing_data[testing_data['session'] == (i+1)])
    end_timestep = testing_sessions[i]['lap_timestep'].max()
    testing_sessions_end.append(testing_sessions[i][testing_sessions[i]['lap_timestep'] == end_timestep])
test_end = pd.concat(testing_sessions_end)
test_end

# create subsets
subsample = test_end[test_end['session'] < 21]
result_success = subsample[subsample['success'] == True]
result_failure = subsample[subsample['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = subsample['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, subsample['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(subsample['session'], np.ones(subsample['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('{} Test Session Results (session=20)'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

# create subsets
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
laptimestep = result_success['lap_timestep'].min()
minacte = result_success['acte'].min()
selected = result_success[result_success['acte'] == minacte]

N = test_end['session'].max()
ind = np.arange(N)  # the x locations for the groups
width = 0.5           # the width of the bars

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['lap_timestep'], width, color='b')
p1 = ax1.plot(result_failure['session'], result_failure['lap_timestep'], 'ro')
p2 = ax1.plot(test_end['session'], np.ones(test_end['session'].count())*laptimestep, 'g')
p3 = ax1.plot(result_success['session'], result_success['acte'], 'y')
p4 = ax1.plot(selected['session'], selected['acte'], 'go')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Time Steps/ACTE", fontsize=10)
plt.title('{} Test Session Results'.format(generation))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Failed Session', 'Completed Lap', 'Accumulated CTE', 'Selected g=n+1'))
plt.show()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, test_end['mse'], width, color='b', label='Session Steering MSE')
p1 = ax1.plot(selected['session'], selected['mse'], 'go', label='Selected g=n+1')
ax1.set_xlabel("Test Session", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('{} Steering MSE Test Session Results'.format(generation))
plt.legend((p0[0], p1[0]), ('Session Steering MSE', 'Selected g=n+1'))
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('{} Session Failure Traces on Test Track Map (session=20)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    if j < 21:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('{} Session Failure Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_failure['session']:
    # corrupted session 38 - skip
    if j != 38:
        testing_sessions.append(testing_data[testing_data['session'] == j])
        testing_sessions_end.append(result_failure[result_failure['session'] == j])
        plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
        sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
        labels.append('Session {} End'.format(j))
        i += 1
plt.legend(sessions, labels)
plt.show()

result_success.describe()

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('Generation {} Session Success Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()
all_result_success = pd.concat([all_result_success, result_success])

p0 = plt.imshow(img, extent=(-338, 315, -213, 203), alpha=0.2)
p1 = plt.plot(x_waypoint, y_waypoint, 'go', ms=3.0)
p2 = plt.plot(x_start, y_start, 'ro', ms=5.0)
testing_sessions = []
testing_sessions_end = []
sessions = [p1[0], p2[0]]
labels = ['Provided Waypoints', 'Starting Position']
plt.title('{} Selected Session Traces on Test Track Map'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
j = int(selected['session'])
testing_sessions.append(testing_data[testing_data['session'] == j])
sessions.append(plt.plot(testing_sessions[0]['x'], testing_sessions[0]['y'], alpha=1.0, lw=0.5, zorder=40-i)[0])
labels.append('Selected Session {}'.format(j))
plt.legend(sessions, labels)
plt.show()

all_result_success.describe()

N = all_result_success['mse'].count()
ind = np.arange(N)  # the x locations for the groups
width = 0.5         # the width of the bars
mse_mean = all_result_success['mse'].apply(np.sqrt).mean()
mse_median = all_result_success['mse'].apply(np.sqrt).median()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
p0 = ax1.bar(ind+1, all_result_success['mse'].apply(np.sqrt), width, color='b')
p1 = ax1.plot(ind+1, np.ones(N)*mse_mean, 'r')
p2 = ax1.plot(ind+1, np.ones(N)*mse_median, 'g')
ax1.set_xlabel("Successful Test Sessions", fontsize=10)
ax1.set_ylabel("Steering Mean Square Error", fontsize=10)
plt.title('All Successful Session Steering MSE Test Session Results')
plt.legend([p0[0], p1[0], p2[0]], ['Session Steering RMSE', 'Mean of RMSE = {}'.format(mse_mean), 'Median of RMSE = {}'.format(mse_median)])
plt.show()



