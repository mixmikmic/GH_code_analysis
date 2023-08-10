import numpy as np


# Initial state 
x = np.matrix([[0.0], 
               [0.0]]) 

# Initial process covariance
P = np.matrix([[1000.0, 0.0], 
               [0.0,    1000.0]]) 

# State function (e.g. time step is one)
F = np.matrix([[1.0, 1.0], 
               [0.0, 1.0]])

# Measurement function
H = np.matrix([[1.0, 0.0]]) 

# Measurement uncertainty
R = np.matrix([[10.0]])

# Process uncertainty
Q = np.matrix([[0.0],
               [0.0]])

# Identity matrix
I = np.matrix([[1.0, 0.0], 
               [0.0, 1.0]]) 

def filter_step(x, P, m):
    # Prediction
    x = F * x             # A priori state (u is zero in this case)
    P = F * P * F.T + Q   # A priori covariance
    
    # Measurement:
    z = np.matrix(m)      # Measurement
    y = z - H * x         # Innovation residual
    S = H * P * H.T + R   # Innovation covariance
    K = P * H.T * S.I     # Optimal Kalman gain
    x = x + K * y         # A posteriori state
    P = (I - K * H) * P   # A posteriori covariance
    
    # NIS:
    n = y.T * S.I * y
    
    return x, P, n

import math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.figure(figsize=(12,12))

# Define our support 
support = range(200)

# Make some noisy measurements
measurements = []
ground_truth = []
for i in support:
    gt = float(i) / 2.0
    ground_truth.append(gt)
    noisy = np.random.normal(gt, 5.0)
    measurements.append(noisy)

# Run the filter and collect ground truth and filter values
filter_vals = []
NIS = []
for z in measurements:
    x, P, n = filter_step(x, P, z)
    filter_vals.append(x[0].getA1()[0])
    NIS.append(n.getA1()[0])
    
# Plot it
plt.plot(support, ground_truth, color='g')
plt.plot(support, measurements, color='r', alpha=0.25)
plt.plot(support, filter_vals, color='b')
plt.show()

total = 0.0
for i in range(len(ground_truth)):
    gt = ground_truth[i]
    kf = filter_vals[i]
    total += (gt - kf)**2
    
rmse = math.sqrt(total / len(ground_truth))

print(rmse)

plt.figure(figsize=(12,12))

plt.plot(support, NIS, color='r', alpha=0.3)
plt.show()



