import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pylab import *

x = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
     2.875,2.875,2.875,2.875,2.875,2.875,2.875,2.875, 
     .22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,
    .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068, .068]
plt.hist(x, bins=288, orientation='horizontal') 
plt.axis([0, 288, 25, 0])

plt.axis([0, 100, 0, 50]) 
plt.hist([0, 1, 4, 12, 40, 96], bins=[37, 37, 35, 25, 20])

val = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
     2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
     2.875,2.875,2.875,2.875,2.875,2.875,2.875,2.875, 
     .22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,.22,
    .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068,.068,.068,.068,.068,.068,.068,.068,.068,.068,
     .068, .068]    # the bar lengths
pos = np.arange(5)+.5    # the bar centers on the y axis

figure(1)
barh(val, width=321)
xlabel('# of Respondents')
title('Immediacy of attempt')
grid(True)

show()

people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel('Performance')
plt.title('How fast do you want to go today?')

plt.show()





from pylab import *
val = [20, 25, 35, 37, 37]    # the bar lengths
pos = arange(5)+.5    # the bar centers on the y axis

figure(1)
barh(pos,val, align='center')
yticks(pos, ( '1 or more days', '2-8 hours', '20 minutes - 1 hour', '5-19 minutes','0-5 minutes'))
xlabel('# of Respondents')
title('Immediacy of attempt')
grid(True)

show()

#plt.axis([0, 100, 0, 50]) 
#[37, 37, 35, 25, 20]
plt.hist([37, 37, 35, 25, 20], bins=5)


#elapsed = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,5,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,13,14,24,24,24,24,34,34,34,34,34,34,34,34,34,34,41,42,43,44,45,46,47,48,49,50,51,52,53]
plt.hist(elapsed)
plt.title("Elapsed")
plt.xlabel("Time")
plt.ylabel("People")
plt.show()
plt.hist(elapsed, bins=[0, 1, 4, 12, 40, 96])



plt.hist(elapsed, bins=range(min(elapsed), max(elapsed) + 5, 5))

plt.figure(elapsed)  
plt.axis([0, 100, 0, 288])  

binBoundaries = np.linspace(0,288,5)

plt.title('A Histogram')  
plt.xlabel('x-axis')  
plt.ylabel('y-axis')  
plt.legend()






x = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,6,7,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,13,14,24,24,24,24,34,34,34,34,34,34,34,34,34,34,41,42,43,44,45,46,47,48,49,50,51,52,53]
plt.hist(x, bins=288) 



