import pandas as pd
df = pd.read_csv('Terrorism Project - Raw Data.csv')
df = df[df['Incident Country'] == 'China']
df = df[['Report Year', 'Confirmed Death(s)', 'Weapon']]

import numpy as np
df['Confirmed Death(s)'] = np.where(df['Confirmed Death(s)'].isnull(), 0, df['Confirmed Death(s)'])
df = df.dropna()
df.head()

deaths_by_weapon = df[['Report Year', 'Weapon', 'Confirmed Death(s)']]
deaths_by_weapon = deaths_by_weapon.groupby(['Report Year', 'Weapon']).agg(['sum'])
# deaths_by_weapon.columns = deaths.columns.get_level_values(0)
# deaths_by_weapon.columns = ['Total Deaths']
# deaths_by_weapon.reset_index(inplace=True)
#################################################################
deaths_by_weapon

deaths = df[['Report Year', 'Confirmed Death(s)']]
deaths = deaths.groupby(['Report Year']).agg(['sum'])
deaths.columns = deaths.columns.get_level_values(0)
deaths.columns = ['Total Deaths']
deaths.reset_index(inplace=True)
#################################################################
working_df = df.copy()
working_df['Incidents with Explosives'] = np.where(working_df['Weapon'].str.contains("Explosives"), 1, 0)
working_df['Incidents with Firearms'] = np.where(working_df['Weapon'].str.contains("Firearms"), 1, 0)
working_df['Incidents with Other Weapons'] = np.where(working_df['Weapon'].str.contains("Other"), 1, 0)
working_df = working_df[['Report Year', 'Incidents with Explosives', 
                         'Incidents with Firearms', 'Incidents with Other Weapons']]
working_df = working_df.groupby(['Report Year']).agg(['sum'])
working_df.columns = working_df.columns.get_level_values(0)
working_df.reset_index(inplace=True)
#################################################################
fatalities_df = df.copy()
fatalities_df['Explosives_Used'] = np.where(fatalities_df['Weapon'].str.contains("Explosives"), 1, 0)
fatalities_df['Firearms_Used'] = np.where(fatalities_df['Weapon'].str.contains("Firearms"), 1, 0)
fatalities_df['Other_Weapons_Used'] = np.where(fatalities_df['Weapon'].str.contains("Other"), 1, 0)
fatalities_df = fatalities_df[['Report Year', 'Confirmed Death(s)', 'Explosives_Used', 
                               'Firearms_Used', 'Other_Weapons_Used']]
#################################################################
explosives = fatalities_df[fatalities_df['Explosives_Used'] == 1]
explosives = explosives[['Report Year', 'Confirmed Death(s)']]
explosives = explosives.groupby(['Report Year']).agg(['sum'])
explosives.columns = explosives.columns.get_level_values(0)
explosives.columns = ['Deaths by Explosives']
explosives.reset_index(inplace=True)
#################################################################
firearms = fatalities_df[fatalities_df['Firearms_Used'] == 1]
firearms = firearms[['Report Year', 'Confirmed Death(s)']]
firearms = firearms.groupby(['Report Year']).agg(['sum'])
firearms.columns = firearms.columns.get_level_values(0)
firearms.columns = ['Deaths by Firearms']
firearms.reset_index(inplace=True)
#################################################################
other_weapons = fatalities_df[fatalities_df['Other_Weapons_Used'] == 1]
other_weapons = other_weapons[['Report Year', 'Confirmed Death(s)']]
other_weapons = other_weapons.groupby(['Report Year']).agg(['sum'])
other_weapons.columns = other_weapons.columns.get_level_values(0)
other_weapons.columns = ['Deaths by Other Weapons']
other_weapons.reset_index(inplace=True)

a = pd.merge(working_df, explosives, how='outer')
b = pd.merge(a, firearms, how='outer')
c = pd.merge(b, other_weapons, how='outer')
d = pd.merge(c, deaths, how='outer')
d['Deaths by Other Weapons'] = np.where(d['Deaths by Other Weapons'].isnull(), 0, d['Deaths by Other Weapons'])
d['Deaths by Firearms'] = np.where(d['Deaths by Firearms'].isnull(), 0, d['Deaths by Firearms'])
d['Deaths by Explosives'] = np.where(d['Deaths by Explosives'].isnull(), 0, d['Deaths by Explosives'])
penultimate_df = d[d['Report Year'] < 1947.0]
penultimate_df

penultimate_df['Mean Fatalities with Explosives'] = penultimate_df['Deaths by Explosives']/                                                     penultimate_df['Incidents with Explosives']
penultimate_df['Mean Fatalities with Firearms']   = penultimate_df['Deaths by Firearms']/                                                     penultimate_df['Incidents with Firearms']
penultimate_df['Mean Fatalities with Other Weapons']   = penultimate_df['Deaths by Other Weapons']/                                                          penultimate_df['Incidents with Other Weapons']
penultimate_df['% Deaths Involving Explosives'] =        (penultimate_df['Deaths by Explosives']/                                                     penultimate_df['Total Deaths'])*100
penultimate_df['% Deaths Involving Firearms'] =          (penultimate_df['Deaths by Firearms']/                                                     penultimate_df['Total Deaths'])*100
penultimate_df['% Deaths Involving Other Weapons'] =     (penultimate_df['Deaths by Other Weapons']/                                                     penultimate_df['Total Deaths'])*100
penultimate_df

final_df = penultimate_df[['Report Year', '% Deaths Involving Explosives', 
                           '% Deaths Involving Firearms', '% Deaths Involving Other Weapons']]

final_df['% Deaths Involving Explosives'] = final_df['% Deaths Involving Explosives'].apply(lambda x: round(x,1))
final_df['% Deaths Involving Firearms'] = final_df['% Deaths Involving Firearms'].apply(lambda x: round(x,1))
final_df['% Deaths Involving Other Weapons'] = final_df['% Deaths Involving Other Weapons'].apply(lambda x: round(x,1))
final_df

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

year    = [1,2,3,4]
labels  = ['a', '1938', '1939', '1940', '1941']
lengend_labels = ['Firearms', 'Explosive Devices']

firearms      = final_df['% Deaths Involving Firearms'].tolist()
explosives   = final_df['% Deaths Involving Explosives'].tolist()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(111)

ax1.plot(year, firearms, linewidth=4, color ='#668cff', marker='o', markersize=20)
ax1.plot(year, explosives, linewidth=4, color ='#ff1a1a',  marker='o', markersize=20)

# plt.xlabel('Year', fontsize=30)
plt.ylabel('% of Fatalities', fontsize=32)

plt.title('Percentage of Fatalities by Weapon Type', fontsize=38, y=1.0)

ax1.set_ylim(0, 100)
plt.yticks(fontsize=30)
ax1.yaxis.set_major_locator(MaxNLocator(6))
ylabels = ['0%', '20%','40%','60%','80%', '100%']
ax1.set_yticklabels(ylabels)

ax1.xaxis.set_major_locator(MaxNLocator(4))
plt.xticks(rotation=25, fontsize=30)
ax1.set_xticklabels(labels)

ax1.grid(True)
ax1.legend(lengend_labels, fontsize=26, loc=2)

# plt.show()
plt.savefig('./images/fatalites_by_weapon_type_line_plot.png')
plt.clf()

