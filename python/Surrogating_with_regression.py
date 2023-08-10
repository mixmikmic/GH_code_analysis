get_ipython().magic('pylab inline')
import pysd
import numpy as np
import pandas as pd

model = pysd.read_vensim('../../models/Manufacturing_Defects/Defects.mdl')

data = pd.read_csv('../../data/Defects_Synthetic/Manufacturing_Defects_Synthetic_Data.csv')
data.head()

plt.scatter(data['Workday'], data['Time per Task'], c=data['Defect Rate'], linewidth=0, alpha=.6)
plt.ylabel('Time per Task')
plt.xlabel('Length of Workday')
plt.xlim(0.15, .45)
plt.ylim(.01, .09)
plt.box('off')
plt.colorbar()
plt.title('Defect Rate Measurements')
plt.figtext(.88, .5, 'Defect Rate', rotation=90, verticalalignment='center');

from sklearn.svm import SVR

Factors = data[['Workday','Time per Task']].values
Outcome = data['Defect Rate'].values
regression = SVR()
regression.fit(Factors, Outcome)

def new_defect_function():
    """ Replaces the original defects equation with a regression model"""
    workday = model.components.length_of_workday()
    time_per_task = model.components.time_allocated_per_unit()
    return regression.predict([workday, time_per_task])[0]

model.components.defect_rate = new_defect_function

model.components.defect_rate()

model.run().plot();



