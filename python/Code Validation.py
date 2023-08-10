get_ipython().magic('matplotlib inline')

D_SAMPLING_FREQUENCY = 500

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('ggplot')

def getChannelData(iChannel, strTestCase):
    cwd = os.getcwd()
    cwd = cwd+'\\CodeValidationData\\'+strTestCase

    f = []
    for (dirpath, dirnames, filenames) in os.walk(cwd):
        f.extend(filenames)
        break
        
    strFileSearch = 'Trace0' + str(iChannel)
    strFiles = filter(lambda x:strFileSearch in x, f)
    
    
    for idx in range(0, len(strFiles)):
        fh = open(cwd+'\\'+strFiles[idx], 'rb')
        # read the data into numpy
        if(idx==0):
            xEnd = np.fromfile(fh, dtype=('>f'))
        else:
            xEnd = np.append(x, np.fromfile(fh, dtype=('>f')))
        fh.close()
    
    # We have to switch the underlying NumPy array to native system
    # Great write up at: http://pandas.pydata.org/pandas-docs/stable/gotchas.html. 
    # If you don't do this you get an error: ValueError: Big-endian buffer not supported on little-endian compiler
    x = xEnd.byteswap().newbyteorder()
    
    return (x,strFiles)

def getDataAsFrame(strTestCase):

    # Read the data in
    (x1,strFiles1) = getChannelData(1,strFolder)
    (x2,strFiles2) = getChannelData(2,strFolder)
    (x3,strFiles3) = getChannelData(3,strFolder)
    (x4,strFiles4) = getChannelData(4,strFolder)
    t = np.divide(range(0,len(x1)),float(D_SAMPLING_FREQUENCY) )

    # Construct the data frame
    dfData = pd.DataFrame(data={('t'):t,
                                ('XAcc'):x1, 
                                ('YAcc'):x2, 
                                ('ZAcc'):x3, 
                                ('Light'):x4,
                                'Surface':strTestCase})
    
    return dfData

def appendDataAsFrame(strTestCase, dfData):
    dfNew = getDataAsFrame(strTestCase)
    dfDataOut = dfData.append(dfNew)
    dfDataOut = dfDataOut.reset_index(drop=True)

    return dfDataOut

def plotFolder(dfDataPlot, strClass):
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(wspace=.5, hspace=0.5)
    
    strColName = 'XAcc'
    ax = dfDataPlot.plot(x='t', y=strColName, 
                     ax=axes[0,0], legend=True, figsize=(10,10))
    ax.set_xlabel('Time, seconds')
    ax.set_ylabel('Amplitude, ADC counts')
    ax.set_title(strColName+'_'+strClass)

    strColName = 'YAcc'
    ax = dfDataPlot.plot(x='t', y=strColName, 
                     ax=axes[0,1], legend=True, figsize=(10,10))
    ax.set_xlabel('Time, seconds')
    ax.set_ylabel('Amplitude, ADC counts')
    ax.set_title(strColName+'_'+strClass)

    strColName = 'ZAcc'
    ax = dfDataPlot.plot(x='t', y=strColName, 
                     ax=axes[1,0], legend=True, figsize=(10,10))
    ax.set_xlabel('Time, seconds')
    ax.set_ylabel('Amplitude, ADC counts')
    ax.set_title(strColName+'_'+strClass)

    strColName = 'Light'
    ax = dfDataPlot.plot(x='t', y=strColName, 
                     ax=axes[1,1], legend=True, figsize=(10,10))
    ax.set_xlabel('Time, seconds')
    ax.set_ylabel('Amplitude, ADC counts')
    ax.set_title(strColName+'_'+strClass)

strFolder = '170331_003'
dfData = getDataAsFrame(strFolder)
#writer = pd.ExcelWriter('output.xlsx')
#dfData.to_excel(writer,sheet_name='Test')

plotFolder(dfData.loc[dfData['Surface'] == strFolder], strFolder)

dfData.head(10)



