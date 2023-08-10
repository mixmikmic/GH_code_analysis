import sys, serial, argparse
import numpy as np
from time import sleep
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
get_ipython().run_line_magic('matplotlib', 'inline')

strPort = '/dev/cu.usbserial-DQ0058HD'     # serial port to access
ser = serial.Serial(strPort, 115200)       # initialise

line = ser.readline()                      # read a line

y_data = float(line.decode('ascii').rstrip()) # decode and strip EOL characters
y_data

x_data = datetime.now().strftime('%H:%M:%S')  # get the timestamp of the reading
x_data

maxLen = 100 # number of readings to display

ax = deque([0.0]*maxLen)    # init the deque of zeros

ax.appendleft(1)     # append to the top of the deque, increases the length of the deque

ax[0]           #

ax[-1]  

len(ax)

ax.pop()          # pop the bottom value, reduces length of deque

len(ax)

def get_data(samples):
    x_list = []
    y_list = []
    for i in range(samples):
        x_data = datetime.now()
        x_list.append(x_data)
        line = ser.readline() 
        y_data = float(line.decode('ascii').rstrip())
        y_list.append(y_data)
        sleep(1)
    return x_list, y_list     

x_list, y_list = get_data(samples=15)

plt.plot(x_list, y_list)



