import sys
#change to match where the PmagPy folder is on your computer
sys.path.insert(0, '/Users/Laurentia/PmagPy')
import pmag,pmagplotlib,ipmag # import PmagPy functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
get_ipython().magic('matplotlib inline')

#strike_dip = [[115,74],[112,63],[110,70],[107,72],[109,68]]
strike_dip = [[142,43],[141,44],[146,38],[157,44]]

bedding_data = pd.DataFrame(strike_dip,columns=['strike','dip'])

bedding_data['pole_trend'] = bedding_data['strike']-90
bedding_data['pole_plunge'] = 90 - bedding_data['dip']
bedding_data

bedding_poles = ipmag.make_di_block(bedding_data['pole_trend'],bedding_data['pole_plunge'])
bedding_poles_mean = pmag.fisher_mean(bedding_poles)
bedding_poles_mean

bedding_poles_mean['dec']

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='stereonet')

mean_strike = bedding_poles_mean['dec'] + 90.0
mean_dip = 90.0 - bedding_poles_mean['inc']

ax.plane(bedding_data['strike'],bedding_data['dip'], 'g-', linewidth=1.5)
ax.pole(bedding_data['strike'],bedding_data['dip'], 'bo', markersize=8)
ax.plane(mean_strike,mean_dip, 'r-', linewidth=3)
ax.pole(mean_strike,mean_dip, 'r^', markersize=15)
ax.grid()

plt.show()
print mean_strike,mean_dip



pmag.dotilt(356.7,42.1,125,61)





