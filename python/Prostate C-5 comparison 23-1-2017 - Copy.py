get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyEclipseDVH import eclipse_DVH
from scipy import interpolate

Conf = eclipse_DVH('Prostate_data_C-5/DVH_conf.txt')
IMRT = eclipse_DVH('Prostate_data_C-5/DVH_IMRT.txt')

IMRT.DVH_df.columns

# IMRT.DVH_df.plot(legend=False)

width=7
height=4
plt.figure(figsize=(width, height))

structure = 'PTVp'
plt.plot(Conf.DVH_df[structure], label="PTV Conf", color='b', ls='--')
plt.plot(IMRT.DVH_df[structure], label="PTV IMRT",  color='b' )

structure = 'Rectum'
plt.plot(Conf.DVH_df[structure], label="Rectum Conf", color='r', ls='--')
plt.plot(IMRT.DVH_df[structure], label="Rectum IMRT",  color='r' )

structure = 'Bladder'
plt.plot(Conf.DVH_df[structure], label="Bladder Conf", color='y', ls='--')
plt.plot(IMRT.DVH_df[structure], label="Bladder IMRT",  color='y' )

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of DVH of Conformal and IMRT plans')
plt.xlabel('Dose (Gy)')
plt.ylabel('Ratio of total structure volume (%)')
plt.ylim([0,105])

Comparison_df = pd.concat([Conf.metrics_df[structure], IMRT.metrics_df[structure]], axis=1)
Comparison_df

def get_Dmetric(df, metric_pct):   # for D50% pass 50
    indexes = np.array(df.index)   # get test data index and values
    values = np.array(df.values)
    f = interpolate.interp1d(values, indexes)  # create the interp object
    return f(metric_pct)

print('The D50% dose is: {} Gy'.format(get_Dmetric(IMRT.DVH_df['PTVp'], 50.0)))

def get_HI(df):   # for D50% pass 50
    indexes = np.array(df.index)   # get test data index and values
    values = np.array(df.values)
    f = interpolate.interp1d(values, indexes)  # create the interp object
    HI = (f(2.0) - f(98.0))/f(50.0)
    return HI 

print('The IMRT HI is: {}'.format(get_HI(IMRT.DVH_df['PTVp'])))

print('The conf HI is: {}'.format(get_HI(Conf.DVH_df['PTVp'])))



