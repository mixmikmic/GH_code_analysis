get_ipython().magic('matplotlib notebook')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from control import *

# Plant
num = [1]
den = [1, 10, 20]
G = tf(num, den)
H = [1]
w, mag, phase = bode(G)

# Plant Step Response
t, s = step_response(G)
target = np.linspace(1, 1, num=len(t))
plt.figure()
plt.plot(t,s,t,target)
plt.ylim([0,1.5])
plt.ylabel(r'Magnitude')
plt.xlabel(r'Time (sec)')
plt.grid()
plt.show

#PID Controller
Kp = 30
Ki = 70
Kd = 1

pid_num = [Kd, Kp, Ki]
pid_den = [1, 0]
C = tf(pid_num,pid_den)

# System
T = feedback(C*G,1)

# PID compensated Step Response
t, s = step_response(T)
target = np.linspace(1, 1, num=len(t))
plt.figure()
plt.plot(t,s,t,target)
plt.ylim([0,1.5])
plt.ylabel(r'Magnitude')
plt.xlabel(r'Time (sec)')
plt.grid()
plt.show



