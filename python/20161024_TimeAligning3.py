import numpy as np
import pandas as pd
from datetime import datetime
import random
mC = np.array([[0.03,0.97,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.17,0.17,0.66,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.11,0.42,0.47,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.06,0.70,0.24,0.00,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.03,0.87,0.10,0.00,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.03,0.94,0.03,0.00,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.22,0.73,0.05,0.00,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.38,0.52,0.10,0.00,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.53,0.33,0.14,0.00],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.66,0.18,0.16],
                [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.91,0.09]])
mCS = np.cumsum(mC,axis=1)
#rawArray will contain the final output for three iterations (variables)
rawArray = []
for nn in range(3):
    #The code from Part 2 is repeated. However, we will not be saving the
    #"true" signal values, only the "raw" signal values.
    curState = 5
    trueStates = [5]
    for ii in range(100000):
        rn = random.random()
        for jj in range(11):
            #Check if the random number is less than the cumulative 
            #probability at each state
            if (rn < mCS[curState,jj]):
                curState = jj
                break
        trueStates.append(curState)
    prevState = 5
    #rawStates is now a dict with the key being the timestamp
    rawStates = {datetime.fromtimestamp(0) : 5}
    #The difference in the following code is that "None" values are not
    #saved anymore, and each value is assigned a timestamp within the dict
    for ii in range(1,len(trueStates)):
        ts = trueStates[ii]
        if (prevState != ts):
            rawStates[datetime.fromtimestamp(ii-1)] = prevState
            rawStates[datetime.fromtimestamp(ii)] = ts
            prevState = ts
    #The results are three pandas Series with timestamps as the index
    rawArray.append(pd.Series(data=rawStates,name="Var" + str(nn)))

#Create a pandas DataFrame from the first variable
df = pd.DataFrame(rawArray[0])
for ii in range(1,3):
    df = pd.merge(left=df,right=pd.DataFrame(rawArray[ii]),
                  left_index=True,right_index=True,how="outer")
df.head(15)

#Percentage decrease formula is (Original - New)/Original
perc_dec = (len(df) - len(df.dropna()))/ len(df)
print("The percentage decrease is: {}%".format(round(perc_dec*100,1)))

df_rs = df.resample("1S").fillna(method="ffill")
df_rs.head(15)

df_ff = df_rs.ffill()
df_ff.head(15)

