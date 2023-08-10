import numpy as np
import pandas as pd

#generate synthetic data
Factors = []
Outcome = []
numpoints = 2000
for workday, time_per_task  in zip(np.random.normal(loc=.3, scale=.05, size=numpoints), np.random.normal(loc=.05, scale=.01, size=numpoints)):
    Factors.append([workday, time_per_task])
    Outcome.append( 0*workday**2/(time_per_task**2) + 1/time_per_task**1.5 + 1000*workday**1.5)

data = pd.DataFrame(Factors, columns=['Workday', 'Time per Task'])
data['Defect Rate'] = Outcome
data['Defect Rate']/= data['Defect Rate'].max()*10
data['Defect Rate'] += np.random.normal(scale=.003, size=len(data['Defect Rate']))
data.head()

data.to_csv('Manufacturing_Defects_Synthetic_Data.csv')



