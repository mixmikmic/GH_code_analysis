import pandas as pd
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15,7) 

# This will give us an error!
data = pd.read_csv('Data/nyc/rest-insp.csv', sep=',')

data = pd.read_csv('Data/nyc/rest-insp.csv', sep=',', encoding='latin1', index_col='Index', parse_dates=['INSPECTION DATE'], dayfirst=False)

data.head(5)

data.tail(5)

data['VIOLATION DESCRIPTION']

data['VIOLATION DESCRIPTION'][:10]

data[['VIOLATION DESCRIPTION','BORO']]

data['VIOLATION DESCRIPTION'].value_counts()

#plt.style.use('ggplot')
viol_counts = data['VIOLATION DESCRIPTION'].value_counts()
viol_counts[:10]
viol_plot = viol_counts[:10].plot(kind='bar')
viol_plot.set_ylabel('No. of Violations')
viol_plot.set_title('NYC Restaurant Inspections: Top 10 Violations')
viol_plot.set_xticklabels( ('Non-food contact surface improperly constructed','Facility not vermin proof','Evidence of mice or live mice','Food contact surface not properly sanitized after use','Cold food item held above 41F', 'Plumbing/sewage disposal system in disrepair','Food not protected from contamination','Sewage-associated (FRSA) flies present in facility','Hot food item not held at or above 140F','Food is adulterated, contaminated, cross-contaminated') )

viol_counts = data['VIOLATION DESCRIPTION'].value_counts()
viol_counts[:10]

#data['VIOLATION DESCRIPTION']
boros = pd.pivot_table(data,index=['BORO'],values=['VIOLATION DESCRIPTION'], aggfunc='count').sort_values(['VIOLATION DESCRIPTION'], ascending=False)
boros

boros.head(5).plot(kind="bar")

pd.pivot_table(data,index=['BORO'],values=['VIOLATION DESCRIPTION'], aggfunc='count').plot(kind="bar")

is_rats = data['VIOLATION DESCRIPTION'] == "Evidence of mice or live mice present in facility's food and/or non-food areas."

rat_complaints = data[is_rats]
rat_complaints['BORO'].value_counts()

rat_counts = rat_complaints['BORO'].value_counts()
rat_plot = rat_counts[:5].plot(kind='bar')
rat_plot.set_ylabel('No. of Violations')
rat_plot.set_title('"Evidence of Mice or Live Mice" Violations by Borough')

data['INSPECTION TYPE'].value_counts()

data['INSPECTION TYPE'].value_counts().sort_values(ascending=True).plot(kind="barh")

