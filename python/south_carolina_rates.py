import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (14.0, 5.0)
matplotlib.rcParams['axes.titlesize'] = 18

def changeColNames(df, year):
    cols = {'stddev_samp(IndividualRate)': 'std_IndividualRate' + '_' + year, 'avg_IndividualRate': 'avg_IndividualRate' + '_' + year}
    return df.rename(columns=cols)

av14 = changeColNames(pd.read_csv('output/averages-2014.csv'), '2014')
av15 = changeColNames(pd.read_csv('output/averages-2015.csv'), '2015')
av16 = changeColNames(pd.read_csv('output/averages-2016.csv'), '2016')
av17 = changeColNames(pd.read_csv('output/averages-2017.csv'), '2017')

mergeCols = ['Age', 'MetalLevel', 'StateCode']
avAll = (av14.merge(av15, on=mergeCols, how='outer')
         .merge(av16, on=mergeCols, how='outer')
         .merge(av17, on=mergeCols, how='outer'))

sc = avAll[(avAll.StateCode == 'SC') & avAll.Age.isin(['25', '35', '45', '55', '65 and over'])]

sc.head()

valColumns = ['avg_IndividualRate_2014', 'avg_IndividualRate_2015','avg_IndividualRate_2016', 'avg_IndividualRate_2017']
scMelted = pd.melt(sc, id_vars=['Age', 'MetalLevel'], value_name='averageRate', var_name='year', value_vars=valColumns)
scMelted.head()

uniqueMetals = scMelted.MetalLevel.unique()
ages = {'25':'red', '35':'blue', '45':'green', '55':'black', '65 and over':'purple'}
fig, axes = plt.subplots(uniqueMetals.shape[0], 1, figsize=(10, 30))
for ax, metal in zip(axes.ravel(), uniqueMetals):
    metalDf = scMelted[scMelted.MetalLevel == metal]
    
    for age, group in metalDf.groupby('Age'):
        group.plot.line(ax=ax, x='year', y='averageRate', label=age, color=ages[age])
    ax.set_title('Plan Level='+metal)
    ax.set_ylim([0, 1500])

uniqueAges = np.sort(scMelted.Age.unique())
metals ={'Catastrophic':'red', 'Platinum':'purple', 'Silver':'silver', 'Gold':'gold', 'Bronze': 'brown'}
fig, axes = plt.subplots(uniqueAges.shape[0], 1, figsize=(10, 30))

for ax, age in zip(axes.ravel(), uniqueAges):
    ageDf = scMelted[scMelted.Age == age]
    for metal, group in ageDf.groupby('MetalLevel'):
        group.plot.line(ax=ax, x='year', y='averageRate', label=metal, color=metals[metal])
    ax.set_title('Age='+age)
    ax.set_ylim([0, 1500])

