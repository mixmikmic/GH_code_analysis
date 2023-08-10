import outliers
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

prescription = pd.read_csv('data/prescriptions_sample.csv.gz', compression='gzip')

medications = outliers.getOverdoseMedications(prescription)
len(medications)

ep_range = np.arange(0.01,1.0,0.01)
results, max_f = outliers.runParameterSearch(prescription, np.asarray(medications), ep_range)

display(results)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8), dpi= 300)

data = []

for i, m in enumerate(max_f.index):
    print(i+1,m)
    data.append(max_f.loc[m].values)
    
plt.boxplot(data)

plt.show()



