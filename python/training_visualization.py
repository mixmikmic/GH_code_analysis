data_file = '../gen5/training-progress.csv'
generation = 'g=5'

import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

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

plt.rcParams["figure.figsize"] = [18, 18]
img=mpimg.imread('track1_top_view.png')

# create subsets
data = pd.read_csv(data_file)
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
result_success = test_end[test_end['success'] == True]
result_failure = test_end[test_end['success'] == False]
result_success.describe()

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
for j in selected['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Selected Session {} End'.format(j))
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
plt.title('Generation {} Session Traces on Test Track Map (session=40)'.format(generation))
plt.xlabel('X')
plt.ylabel('Y')
i = 0
for j in result_success['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_success[result_success['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 's', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Successful Session {}'.format(j))
    i += 1
for j in result_failure['session']:
    testing_sessions.append(testing_data[testing_data['session'] == j])
    testing_sessions_end.append(result_failure[result_failure['session'] == j])
    plt.plot(testing_sessions[i]['x'], testing_sessions[i]['y'], alpha=1.0, lw=0.5, zorder=40-i)
    sessions.append(plt.plot(testing_sessions_end[i]['x'], testing_sessions_end[i]['y'], 'x', alpha=1.0, lw=0.5, zorder=40-i)[0])
    labels.append('Failed Session {} End'.format(j))
    i += 1
plt.legend(sessions, labels)
plt.show()






