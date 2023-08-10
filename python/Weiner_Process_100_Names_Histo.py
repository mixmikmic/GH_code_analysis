import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats  import invgauss
from random import random

mu =  0.00500 
sigma =  0.00500 
dt = 1
S0 = 1


#create a big array of nones to hold our data
w,h = 400, 10
array = [[None] * w for i in range(h)]

#set up the column headings
columns = []
for i in range(100):
    columns.append('Time' + str(i))
    columns.append('density' + str(i))
    columns.append('dS' + str(i))
    columns.append('S' + str(i))

#set up the indexes
index = []
for i  in range (h):
    index.append(str(i))

#create a dataframe based on the columns and indices
df = pd.DataFrame(array, columns=columns, index=index)

#intialize first row
first_row = [0, 0,0,S0] * 100

df.loc[0] = first_row


t1 = dt
density1 = 0.00500 
dS1 = S0 * density1
S1 = S0 + dS1

#initialize the second row
second_row = [t1, density1, dS1, S1] * 100

df.loc[1] = second_row

for i in range(2, 12):
    prev_iter = df.loc[i-1]
    #print(prev_iter)
    
    new_row = []
    
    for j in range(100):
        prev_index = str(j)
        time_hash = "Time" + prev_index
        density_hash = "density" + prev_index 
        dS_hash = "dS" + prev_index
        S_hash = "S" + prev_index
    
        new_time = prev_iter[time_hash] + dt
        randvar = invgauss.rvs(mu=random(), loc=0 ,scale=dt)
        walk = mu * dt + sigma * randvar
        new_density = walk
        new_dS = prev_iter[dS_hash] + new_density
        new_S = prev_iter[S_hash] + new_dS
        
        new_row.append(new_time)
        new_row.append(new_density)
        new_row.append(new_dS)
        new_row.append(new_S)
        
    df.loc[i] = new_row


print(df)

x = df["Time0"]

for i in range(100):
    hash = "S" + str(i)
    plt.plot(x,df[hash], marker='o', linestyle='solid')

plt.title("Share Price - Wiener Stochastic Process - 100 Names")
plt.show()

