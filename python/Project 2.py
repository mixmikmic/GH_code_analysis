#
# Load up some packages
#
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

df=pd.read_csv('triangle.csv', sep='\t')
df[df['time']>0.03].plot('time','V(C)')
pl.grid()
pl.legend(loc=2)
pl.ylabel("Vc (Volts)")
pl.title("Steady State signal at Capacitor")



