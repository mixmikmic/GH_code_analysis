import numpy as np
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(sum(map(ord, "aesthetics")))
get_ipython().magic('matplotlib inline')
import seaborn as sns

data=pd.read_csv('/Users/mkulunyar/Dropbox/NYU/Fall16/DataScience/DataScience_Project/BaseballsAndBooleans_NYU-CDS/DT_MaxDepth20_MinSplit300.csv')
data2=pd.read_csv('/Users/mkulunyar/Dropbox/NYU/Fall16/DataScience/DataScience_Project/BaseballsAndBooleans_NYU-CDS/DT_MaxDepth20_MinSplit300_MinLeaf50.csv')

data.head(4)

plt.plot(data['Prediction_Certainty_Threshold'], data['net_gain'])
plt.title('Net Gain vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Net Gain', fontsize=10)

plt.plot(data['Prediction_Certainty_Threshold'], data['Precision'])
plt.title('Precision vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Precision', fontsize=10)

plt.plot(data['Prediction_Certainty_Threshold'], data['Cumulative Percent of Records'])
plt.title('Cumulative Percent of Records vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Cumulative Percent of Records', fontsize=10)

plt.plot(data2['Prediction_Certainty_Threshold'], data2['net_gain'])
plt.title('Net Gain vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Net Gain', fontsize=10)

plt.plot(data2['Prediction_Certainty_Threshold'], data2['Precision'])
plt.title('Precision vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Precision', fontsize=10)

plt.plot(data2['Prediction_Certainty_Threshold'], data2['Cumulative Percent of Records'])
plt.title('Cumulative Percent of Records vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Cumulative Percent of Records', fontsize=10)

data3=pd.read_csv('/Users/mkulunyar/Dropbox/NYU/Fall16/DataScience/DataScience_Project/BaseballsAndBooleans_NYU-CDS/RF_N200_MaxDepth20_MinSplit300.csv')
data4=pd.read_csv('/Users/mkulunyar/Dropbox/NYU/Fall16/DataScience/DataScience_Project/BaseballsAndBooleans_NYU-CDS/RF_N100_MaxDepth20_MinSplit300.csv')

plt.plot(data3['Prediction_Certainty_Threshold'], data3['net_gain'])
plt.title('Net Gain vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Net Gain', fontsize=10)

plt.plot(data3['Prediction_Certainty_Threshold'], data3['Precision'])
plt.title('Precision vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Precision', fontsize=10)

plt.plot(data3['Prediction_Certainty_Threshold'], data3['Cumulative Percent of Records'])
plt.title('Cumulative Percent of Records vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Cumulative Percent of Records', fontsize=10)

plt.plot(data4['Prediction_Certainty_Threshold'], data4['net_gain'])
plt.title('Net Gain vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Net Gain', fontsize=10)

plt.plot(data4['Prediction_Certainty_Threshold'], data4['Precision'])
plt.title('Precision vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Precision', fontsize=10)

plt.plot(data4['Prediction_Certainty_Threshold'], data4['Cumulative Percent of Records'])
plt.title('Cumulative Percent of Records vs Threshold', fontsize=18)
plt.xlabel('Threshold', fontsize=10)
plt.ylabel('Cumulative Percent of Records', fontsize=10)



