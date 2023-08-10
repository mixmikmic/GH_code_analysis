import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import pandas as pd
import seaborn as sns
import matplotlib as mp
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm

sns.set()

# Read simulation files
def datafileread(measurename,skipfirstrows):
    # Reading Datafiles
    path = measurename
    data = np.genfromtxt(path,
                        skip_header=skipfirstrows,
                        delimiter=',',
                        dtype=(float,float),
                        unpack=True)
    return data

def loadModel(filename, skiprows, decimals=1, useRawInput=False):
    # measurement
    tlp_v,tlp_length,nominal,amplitude,width,input_amplitude = datafileread(filename,skiprows)

    tlp_length = np.round(tlp_length * 1e9, decimals=1)
    width *= 1e9
    
    input_amplitude = np.round(input_amplitude, decimals=decimals)
    
    if useRawInput:
        input_amplitude = tlp_v
    
    print("%d %d" % (len(np.unique(input_amplitude)), len(np.unique(tlp_v))))
    assert len(np.unique(input_amplitude)) == len(np.unique(tlp_v))
    
    print("## Model: %s" % filename)

    
    df_amplitude = pd.DataFrame({'x': tlp_length, 'y': input_amplitude, 'z': amplitude})
    df_amplitude = df_amplitude.pivot(index='y',columns='x', values='z')
    df_amplitude = df_amplitude.dropna()

    df_width = pd.DataFrame({'x': tlp_length, 'y': input_amplitude, 'z': width})
    df_width = df_width.pivot(index='y',columns='x', values='z')
    df_width = df_width.dropna()
    
    print("Width : %E (ns) -> %E (ns)" % (df_width.values.min(), df_width.values.max()))
    print("Amplitude : %E (V) -> %E (V)" % (df_amplitude.values.min(), df_amplitude.values.max()))
    
    return df_amplitude, df_width

def findClosest(dataframe, xin, yin):
    xnearest = dataframe.columns.map(lambda x: abs(xin-x)).argsort()
    ynearest = dataframe.index.map(lambda y: abs(yin-y)).argsort()
    return xnearest[0], ynearest[0]

def findNextPair(df_x, df_y, xin, yin):
    #
    x,y = findClosest(df_x, xin, yin)
    #print(df_x.columns)
    #print("x[%f vs %f] (ns) & y[%f vs %f] (V)" % (xin, df_x.columns[x],yin, df_x.index[y]))
    print("d %f vs %f (ns)" % (xin, df_x.columns[x]))
    print("d %f vs %f (V)" % (yin, df_x.index[y]))
    
    if x <= 0 or x >= len(df_x.columns) - 1:
        print("Warning : x index (%d) at boundary" % x)
        
    if y <= 0 or y >= len(df_x.index) - 1:
        print("Warning : y index (%d) at boundary" % y)
       
        
    outputX = df_x.iloc[y,x]
    
    #
    x,y = findClosest(df_y, xin, yin)
    #print(df_x.index)
    #print("x[%f vs %f] (ns) & y[%f vs %f] (V)" % (xin, df_y.columns[x],yin, df_y.index[y]))
    outputY = df_y.iloc[y,x]
   
        
    if x <= 0 or x >= len(df_y.columns) - 1:
        print("Warning : x index (%d) at boundary" % x)
        #print("%f vs %f (ns)" % (xin, df_y.columns[x]))
       
    
    if y <= 0 or y >= len(df_y.index) - 1:
        print("Warning : y index (%d) at boundary" % y)
        #print("%f vs %f (ns)" % (yin, df_y.index[y]))
        
    
    return outputX, outputY

vpre_df_amp, vpre_df_width = loadModel('cz_vpre_V4_50p.csv', 19, useRawInput=True)
bg_df_amp, bg_df_width = loadModel('cz_bandgap_V4_50p.csv', 14)
reg_df_amp, reg_df_width = loadModel('cz_regulator_V4_50p.csv', 14, decimals=4)

# Input TLP stress characteristics

inputX = 100 # ns
inputY = -50 # V

# Apply the models

pair1 = findNextPair(vpre_df_width, vpre_df_amp, inputX, inputY)
print("Vpre done")
pair2 = findNextPair(bg_df_width, bg_df_amp, pair1[0], pair1[1])
print("Bandgap done")
pair3 = findNextPair(reg_df_width, reg_df_amp, pair2[0], pair2[1])

print("##                       Vpre output values : (%.0f ns, %.2f V)" % pair1)
print("##                    Bandgap output values : (%.0f ns, %.2f V)" % pair2)
print("##                  Regulator output values : (%.0f ns, %.2f V)" % pair3)

