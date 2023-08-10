import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib inline')

# Loading the result datasets

graph_result = pd.read_csv('./Data_Subset/IFResult/graph_result.csv')
user_log_result = pd.read_csv('./Data_Subset/IFResult/user_log_result.csv')
device_file_result = pd.read_csv('./Data_Subset/IFResult/device_file_result.csv')
psychometric_result = pd.read_csv('./Data_Subset/IFResult/psychometric_result.csv')
All_params_result = pd.read_csv('./Data_Subset/IFResult/All_params_result.csv')

# User's anomaly score for all features

f, ax = plt.subplots(figsize = (25,15))
x_col='user'
y_col = 'ascore'

sns.pointplot(ax=ax,x=x_col,y=y_col,data=All_params_result,color='blue')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=graph_result,color='green')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=user_log_result,color='red')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=psychometric_result,color='brown')
sns.pointplot(ax=ax,x=x_col,y=y_col,data=device_file_result,color='orange')


ax.legend(handles=ax.lines[::len(All_params_result)+1], labels=["All","Graph","Logon/Logoff","Psychometric","Removable Media"])

ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])
#plt.gcf().autofmt_xdate()
#ax.set_xtickslabels(rotation = 45)
ax.set_title('Anomaly score for different set of parameters', size = 20)
plt.rcParams["axes.labelsize"] = 25
plt.xticks(rotation = 45, fontsize = 10)
plt.yticks(fontsize = 10)
#plt.show()
plt.legend(loc = 'best')

user_pc_gdegree = pd.read_csv('./Data_Subset/Input_features/user_pc_gdegree.csv')

user_pc_gdegree.head()

# node degree from the graph analysis
plt.hist(user_pc_gdegree.pc_count, alpha=1, color = 'green');
plt.xlabel('Number of PC', size =8)
plt.ylabel('Number of User', size = 8)
plt.title('PC count distribution of users', size=12)

user_pc_gdegree.loc[user_pc_gdegree['pc_count'] > 40]

# files per day stats

files_per_day_stats = pd.read_csv('./Data_Subset/Input_features/files_per_day_stats.csv')

files_per_day_stats.head()


f, ax = plt.subplots(figsize = (25,15))
x_col='user'

sns.pointplot(ax=ax,x=x_col,y='mode',data=files_per_day_stats,color='teal')
sns.pointplot(ax=ax,x=x_col,y='max',data=files_per_day_stats,color='magenta')

ax.legend(handles=ax.lines[::len(files_per_day_stats)+1], labels=["mode","max"])

ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])

ax.set_title('File transfer per day', size = 30)
plt.rcParams["axes.labelsize"] = 25
plt.xticks(rotation = 45, fontsize = 10)
plt.yticks(fontsize = 10)
plt.legend(loc = 'best')



