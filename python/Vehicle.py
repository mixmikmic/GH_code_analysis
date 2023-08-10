import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("Accerometer_data.csv",nrows=2000)

data.head(2)

plt.style.use(['dark_background'])
fig =  plt.figure(figsize=(28,5))
ax1 = fig.add_subplot(121)
ax1.plot(data['Time'], data['Y'] ,'r-', label = 'Y Acc.')
ax1.set_title('Acceleration in the forward (Y) direction')
ax1.set_ylabel('Y-Acc.')
ax1.set_xlabel('Time')

plt.xticks(rotation = 45)
plt.style.use('ggplot')
plt.show()

K=list(data['Y'])[1:]
K.append(0.00)
O=[list(data['Y'])[i]-K[i] for i in range(len(K))]
O=pd.Series(O)
high = data['Y'][(abs(O)>3)].index
high = list(high)

plt.style.use(['dark_background'])
fig =  plt.figure(figsize=(28,5))
ax1 = fig.add_subplot(121)
ax1.plot(data['Time'], data['Y'] ,'-gD', label = 'Points of overspeeding/jerks', markevery= high)
ax1.set_title('Acceleration in the Y direction')
ax1.set_ylabel('Y Acc.')
ax1.set_xlabel('Timestamp')
plt.legend(frameon=False)
plt.xticks(rotation = 45)

plt.show()

high = data['Y'][(abs(data['Y'])>3)].index
high = list(high)
plt.style.use(['dark_background'])
fig =  plt.figure(figsize=(28,5))
ax1 = fig.add_subplot(121)
ax1.plot(data['Time'], data['Y'] ,'-rD', label = 'Points of overspeeding/jerks', markevery= high)
ax1.set_title('Acceleration in the Y direction')
ax1.set_ylabel('Y Acc.')
ax1.set_xlabel('Timestamp')
plt.legend(frameon=False)
plt.xticks(rotation = 45)

plt.show()

#Mean accelration in Z direction
Mean=sum(data['Z'])/len(data['Z'])
Mean

high = data['Z'][(abs(data['Z'])>15)].index # a lot high than normal
high = list(high)
fig =  plt.figure(figsize=(28,5))
ax1 = fig.add_subplot(121)
ax1.plot(data['Time'], data['Z'] ,'-bD', label = 'Potholes/Bumps', markevery= high)
ax1.set_title('Acceleration in the Z direction')
ax1.set_ylabel('Z Acc.')
ax1.set_xlabel('Timestamp')
plt.legend(frameon=False)
plt.xticks(rotation = 45)

plt.show()

plt.style.use(['dark_background'])
K=pd.Series([data['X'].tolist()[:-1][i]-data['X'].tolist()[1:][i] for i in range(len(data['X'].tolist())-1)])
fig =  plt.figure(figsize=(28,5))
ax1 = fig.add_subplot(121)
ax1.plot(data['Time'][0:-1], K ,'r-', label = 'X Acc.')
ax1.set_title('Acceleration in the forward (X) direction')
ax1.set_ylabel('X-Acc.')
ax1.set_xlabel('Time')

plt.xticks(rotation = 45)
plt.style.use('ggplot')
plt.show()



