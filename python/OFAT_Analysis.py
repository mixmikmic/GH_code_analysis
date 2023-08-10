get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import easygui


Data = pd.read_csv("OFATData_1.csv")

Data["Attempts_each"] = Data["Total_attempts"] / Data["N"]

Distribution = 'circle'
Grid = True


# CIRCLE _ OPEN GRID
Pole_data_CO = Data[(Data['N']==250) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==True) & (Data['Vision']==2)]
EV_data_CO = Data[(Data['N_poles']==1/6) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==True) & (Data['Vision']==2)]
Vision_data_CO = Data[(Data['N']==250) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==True) & (Data['N_poles']==1/6)]

# CIRCLE _ CLOSED GRID
Pole_data_CC = Data[(Data['N']==250) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==False) & (Data['Vision']==2)]
EV_data_CC = Data[(Data['N_poles']==1/6) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==False) & (Data['Vision']==2)]
Vision_data_CC = Data[(Data['N']==250) & (Data['Grid_positions']=='circle') & (Data['Grid_open']==False) & (Data['N_poles']==1/6)]

# LHS _ OPEN GRID
Pole_data_LO = Data[(Data['N']==250) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==True) & (Data['Vision']==2)]
EV_data_LO = Data[(Data['N_poles']==1/6) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==True) & (Data['Vision']==2)]
Vision_data_LO = Data[(Data['N']==250) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==True) & (Data['N_poles']==1/6)]

# LHS _ CLOSED GRID
Pole_data_LC = Data[(Data['N']==250) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==False) & (Data['Vision']==2)]
EV_data_LC = Data[(Data['N_poles']==1/6) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==False) & (Data['Vision']==2)]
Vision_data_LC = Data[(Data['N']==250) & (Data['Grid_positions']=='LHS') & (Data['Grid_open']==False) & (Data['N_poles']==1/6)]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")



sns.swarmplot(x='N', y='Usage', data=EV_data_CO)
sns.boxplot(x='N', y='Usage', data=EV_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.14,0.3])
plt.show()

sns.swarmplot(x='N', y='Usage', data=EV_data_CC)
sns.boxplot(x='N', y='Usage', data=EV_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.14,0.3])
plt.show()

sns.swarmplot(x='N', y='Usage', data=EV_data_LO)
sns.boxplot(x='N', y='Usage', data=EV_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.14,0.3])
plt.show()

sns.swarmplot(x='N', y='Usage', data=EV_data_LC)
sns.boxplot(x='N', y='Usage', data=EV_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.14,0.3])
plt.show()

sns.swarmplot(x='N', y='Percentage_failed', data=EV_data_CO)
sns.boxplot(x='N', y='Percentage_failed', data=EV_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.5])
plt.show()

sns.swarmplot(x='N', y='Percentage_failed', data=EV_data_CC)
sns.boxplot(x='N', y='Percentage_failed', data=EV_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.5])
plt.show()

sns.swarmplot(x='N', y='Percentage_failed', data=EV_data_LO)
sns.boxplot(x='N', y='Percentage_failed', data=EV_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.5])
plt.show()

sns.swarmplot(x='N', y='Percentage_failed', data=EV_data_LC)
sns.boxplot(x='N', y='Percentage_failed', data=EV_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.5])
plt.show()

sns.swarmplot(x='N', y='Attempts_each', data=EV_data_CO)
sns.boxplot(x='N', y='Attempts_each', data=EV_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,11])
plt.show()

sns.swarmplot(x='N', y='Attempts_each', data=EV_data_CC)
sns.boxplot(x='N', y='Attempts_each', data=EV_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,11])
plt.show()

sns.swarmplot(x='N', y='Attempts_each', data=EV_data_LO)
sns.boxplot(x='N', y='Attempts_each', data=EV_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,11])
plt.show()

sns.swarmplot(x='N', y='Attempts_each', data=EV_data_LC)
sns.boxplot(x='N', y='Attempts_each', data=EV_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,11])
plt.show()

sns.swarmplot(x='N', y='Average_lifespan', data=EV_data_CO)
sns.boxplot(x='N', y='Average_lifespan', data=EV_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2300,2500])
plt.show()

sns.swarmplot(x='N', y='Average_lifespan', data=EV_data_CC)
sns.boxplot(x='N', y='Average_lifespan', data=EV_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2300,2500])
plt.show()

sns.swarmplot(x='N', y='Average_lifespan', data=EV_data_LO)
sns.boxplot(x='N', y='Average_lifespan', data=EV_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2300,2500])
plt.show()

sns.swarmplot(x='N', y='Average_lifespan', data=EV_data_LC)
sns.boxplot(x='N', y='Average_lifespan', data=EV_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2300,2500])
plt.show()

sns.swarmplot(x='N_poles', y='Usage', data=Pole_data_CO)
sns.boxplot(x='N_poles', y='Usage', data=Pole_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.1,0.35])
plt.show()

sns.swarmplot(x='N_poles', y='Usage', data=Pole_data_CC)
sns.boxplot(x='N_poles', y='Usage', data=Pole_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.1,0.35])
plt.show()

sns.swarmplot(x='N_poles', y='Usage', data=Pole_data_LO)
sns.boxplot(x='N_poles', y='Usage', data=Pole_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.1,0.35])
plt.show()

sns.swarmplot(x='N_poles', y='Usage', data=Pole_data_LC)
sns.boxplot(x='N_poles', y='Usage', data=Pole_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.1,0.35])
plt.show()

sns.swarmplot(x='N_poles', y='Percentage_failed', data=Pole_data_CO)
sns.boxplot(x='N_poles', y='Percentage_failed', data=Pole_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.6])
plt.show()

sns.swarmplot(x='N_poles', y='Percentage_failed', data=Pole_data_CC)
sns.boxplot(x='N_poles', y='Percentage_failed', data=Pole_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.6])
plt.show()

sns.swarmplot(x='N_poles', y='Percentage_failed', data=Pole_data_LO)
sns.boxplot(x='N_poles', y='Percentage_failed', data=Pole_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.6])
plt.show()

sns.swarmplot(x='N_poles', y='Percentage_failed', data=Pole_data_LC)
sns.boxplot(x='N_poles', y='Percentage_failed', data=Pole_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.05,0.6])
plt.show()

sns.swarmplot(x='N_poles', y='Attempts_each', data=Pole_data_CO)
sns.boxplot(x='N_poles', y='Attempts_each', data=Pole_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([5,13])
plt.show()

sns.swarmplot(x='N_poles', y='Attempts_each', data=Pole_data_CC)
sns.boxplot(x='N_poles', y='Attempts_each', data=Pole_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([5,13])
plt.show()

sns.swarmplot(x='N_poles', y='Attempts_each', data=Pole_data_LO)
sns.boxplot(x='N_poles', y='Attempts_each', data=Pole_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([5,13])
plt.show()

sns.swarmplot(x='N_poles', y='Attempts_each', data=Pole_data_LC)
sns.boxplot(x='N_poles', y='Attempts_each', data=Pole_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([5,13])
plt.show()

sns.swarmplot(x='N_poles', y='Average_lifespan', data=Pole_data_CO)
sns.boxplot(x='N_poles', y='Average_lifespan', data=Pole_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2400,2500])
plt.show()

sns.swarmplot(x='N_poles', y='Average_lifespan', data=Pole_data_CC)
sns.boxplot(x='N_poles', y='Average_lifespan', data=Pole_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2400,2500])
plt.show()

sns.swarmplot(x='N_poles', y='Average_lifespan', data=Pole_data_LO)
sns.boxplot(x='N_poles', y='Average_lifespan', data=Pole_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2400,2500])
plt.show()

sns.swarmplot(x='N_poles', y='Average_lifespan', data=Pole_data_LC)
sns.boxplot(x='N_poles', y='Average_lifespan', data=Pole_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2400,2500])
plt.show()

sns.swarmplot(x='Vision', y='Usage', data=Vision_data_CO)
sns.boxplot(x='Vision', y='Usage', data=Vision_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.15,0.22])
plt.show()

sns.swarmplot(x='Vision', y='Usage', data=Vision_data_CC)
sns.boxplot(x='Vision', y='Usage', data=Vision_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.15,0.22])
plt.show()

sns.swarmplot(x='Vision', y='Usage', data=Vision_data_LO)
sns.boxplot(x='Vision', y='Usage', data=Vision_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.15,0.22])
plt.show()

sns.swarmplot(x='Vision', y='Usage', data=Vision_data_LC)
sns.boxplot(x='Vision', y='Usage', data=Vision_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.15,0.22])
plt.show()

sns.swarmplot(x='Vision', y='Percentage_failed', data=Vision_data_CO)
sns.boxplot(x='Vision', y='Percentage_failed', data=Vision_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.08,0.35])
plt.show()

sns.swarmplot(x='Vision', y='Percentage_failed', data=Vision_data_CC)
sns.boxplot(x='Vision', y='Percentage_failed', data=Vision_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.08,0.35])
plt.show()

sns.swarmplot(x='Vision', y='Percentage_failed', data=Vision_data_LO)
sns.boxplot(x='Vision', y='Percentage_failed', data=Vision_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.08,0.35])
plt.show()

sns.swarmplot(x='Vision', y='Percentage_failed', data=Vision_data_LC)
sns.boxplot(x='Vision', y='Percentage_failed', data=Vision_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([0.08,0.35])
plt.show()

sns.swarmplot(x='Vision', y='Attempts_each', data=Vision_data_CO)
sns.boxplot(x='Vision', y='Attempts_each', data=Vision_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,9])
plt.show()

sns.swarmplot(x='Vision', y='Attempts_each', data=Vision_data_CC)
sns.boxplot(x='Vision', y='Attempts_each', data=Vision_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,9])
plt.show()

sns.swarmplot(x='Vision', y='Attempts_each', data=Vision_data_LO)
sns.boxplot(x='Vision', y='Attempts_each', data=Vision_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,9])
plt.show()

sns.swarmplot(x='Vision', y='Attempts_each', data=Vision_data_LC)
sns.boxplot(x='Vision', y='Attempts_each', data=Vision_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([6,9])
plt.show()

sns.swarmplot(x='Vision', y='Average_lifespan', data=Vision_data_CO)
sns.boxplot(x='Vision', y='Average_lifespan', data=Vision_data_CO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2450,2500])
plt.show()

sns.swarmplot(x='Vision', y='Average_lifespan', data=Vision_data_CC)
sns.boxplot(x='Vision', y='Average_lifespan', data=Vision_data_CC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2450,2500])
plt.show()

sns.swarmplot(x='Vision', y='Average_lifespan', data=Vision_data_LO)
sns.boxplot(x='Vision', y='Average_lifespan', data=Vision_data_LO,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2450,2500])
plt.show()

sns.swarmplot(x='Vision', y='Average_lifespan', data=Vision_data_LC)
sns.boxplot(x='Vision', y='Average_lifespan', data=Vision_data_LC,
        showcaps=True,boxprops={'facecolor':'None'},
        showfliers=False,whiskerprops={'linewidth':1})
plt.ylim([2450,2500])
plt.show()

from scipy import stats



f_value, p_value = stats.f_oneway(EV_data_CO['Usage'][EV_data_CO['N']==100],EV_data_CO['Usage'][EV_data_CO['N']==250],EV_data_CO['Usage'][EV_data_CO['N']==400])

f_value,p_value

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey =  pairwise_tukeyhsd(EV_data_CO["Usage"], EV_data_CO["N"])
print(tukey)

tukey =  pairwise_tukeyhsd(Pole_data_CO["Usage"], Pole_data_CO["N_poles"])
print(tukey)


f_value, p_value = stats.f_oneway(Pole_data_CO['Usage'][Pole_data_CO['N_poles']==1/10],Pole_data_CO['Usage'][Pole_data_CO['N_poles']==1/8],Pole_data_CO['Usage'][Pole_data_CO['N_poles']==1/6],Pole_data_CO['Usage'][Pole_data_CO['N_poles']==1/4])

