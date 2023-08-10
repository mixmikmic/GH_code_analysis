import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

data = pd.read_csv('../data/hbv_s_data.csv', index_col=0, parse_dates=True)

evap_true = np.array([0.6,1.9,2.4,1.8,1.4,1.3,1.0,0.8,0.6,0.4,0.2,0.3])*1.2 #evapo for jan-dec

def romanenko(data):
    Ta = np.array([data.ix[data.index.month == x, 'Temp'].mean() for x in range(1, 13)])
    Td = 2
    
    def et(T):
        return 33.8679*( ((0.00738*T + 0.8072)**8) - 0.000019*(np.abs(1.8*T + 48)) + 0.001316)
    
    Rh = et(Td)*100/et(Ta)
    # best results with manual evap setup
    # Rh = np.array([60, 0, 0, 45, 70, 80, 85, 90, 90, 90, 90, 76])
    return (0.0018*((25+Ta)**2))*(100-Rh)/30

Ta = np.array([data.ix[data.index.month == x, 'Temp'].mean() for x in range(1, 13)])
Td = np.array([0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
et(Td)*100/et(Ta)

Ta

plt.plot(range(1,13), romanenko(data), 'b',
         range(1,13), evap_true, 'r')



def kharufa(data):
    Ta = np.array([data.ix[data.index.month == x, 'Temp'].mean() for x in range(1, 13)])
    p = 0.85
    return 0.34*p*(Ta**1.3)

plt.plot(range(1,13), kharufa(data), 'b', 
         range(1,13), romanenko(data), 'g',
         range(1,13), evap_true, 'r')

print(kharufa(data))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../models/')
import hbv_s
swe_riv = pd.read_csv('../data/hbv_s_data.csv', index_col=0, parse_dates=True)
Qobs = swe_riv.Qobs
Qsim = hbv_s.simulation(swe_riv)
pd.DataFrame({'obs':Qobs, 'sim':Qsim}, index=swe_riv.index).plot()

swe_riv2 = swe_riv.drop('Evap', 1)

swe_riv2['Evap'] = swe_riv2.index.map(lambda x: romanenko(swe_riv2)[x.month-1])

Qsim = hbv_s.simulation(swe_riv2)
pd.DataFrame({'obs':Qobs, 'sim':Qsim}, index=swe_riv2.index).plot()

swe_riv2['Evap'].plot();
swe_riv['Evap'].plot()

evap_true

a = np.array([0.6,1.9,2.4,1.8,1.4,1.3,1.0,0.8,0.6,0.4,0.2,0.3])

np.savetxt('../data/Evap_monthly_constants.txt', a)

np.loadtxt('../data/Evap_monthly_constants.txt')

swe_riv2['Evap2'] = swe_riv2.index.map(lambda x: np.loadtxt('../data/Evap_monthly_constants.txt')[x.month-1])

swe_riv2[['Evap', 'Evap2']].plot()



