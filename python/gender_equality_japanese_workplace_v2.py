import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

df = pd.read_csv('data/gender_equality_japanese_workplace_eng.csv', header=1)
total = len(df)

import textwrap

def draw_barchart(x, y, x_ticks, y_ticks, title):
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    
    ax.spines["top"].set_visible(False) 
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_color('#1a1a1a')
    ax.spines['bottom'].set_color('#1a1a1a')
    
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    
    ax.tick_params(axis='x', colors='#1a1a1a')
    ax.tick_params(axis='y', colors='#1a1a1a')
    
    plt.xticks(fontsize=14, color='#1a1a1a')  
    plt.yticks(y_ticks, fontsize=14, color='#1a1a1a') 
    plt.xticks(x, x_ticks)
    
    title = "\n".join(textwrap.wrap(title, 60))
    plt.title(title, fontsize=18, color='#1a1a1a', y=1.1)
    
    return ax.bar(x, y, width= 0.8, color="#ff5733", edgecolor = "none")

df.head()

df.describe()

def response_convert(r):
    if r == 'agree':
        return 1
    elif r == 'disagree':
        return 0
    else:
        return r

df.columns = ['Age', 'Skilled', 'More Women', 'Male Boss', 'Unequal', 'Disadvantaged',
              'Female Tasks', 'Male Tasks', 'Managerial', 'Quit', 'Maternity',
              'Male Maternity', 'Pressure']

# Age is not binary so we need to remove it from the dataframe
# and then put it back after we have ran the conversion function.
age_col = df.pop('Age')
df = df.applymap(response_convert).astype(float)
df['Age'] = pd.Series(age_col, index=df.index)

df.head()

percentage = lambda x, y: x / y * 100

df_equal = df.loc[df['Unequal'] == 0, ['Female Tasks', 'Male Maternity', 'Pressure']]
total_equal = len(df_equal)

print('% Equal: {0}'.format(percentage(total_equal, total)))

df_equal.head()

equal = len(df_equal.loc[(df_equal['Female Tasks'] == 0) & 
       (df_equal['Male Maternity'] == 1) & 
       (df_equal['Pressure'] == 0)])

print("Entries: {0} | Equal: {1} | % True Equal: {2}".format(total_equal, equal, percentage(equal, total_equal)))

equal_tasks_percentage = percentage(df_equal['Female Tasks'].sum(), total_equal)
equal_pressure_percentage = percentage(df_equal['Pressure'].sum(), total_equal)

y = [100, equal_tasks_percentage, equal_pressure_percentage]
x = [0, 1, 2]
x_ticks = ['Equal', 'Female Tasks', 'Pressure']
y_ticks = range(0, 100, 10)
title = 'A Graph to Show Women Who Both Said They Felt Equal and Indicated Unequal Treatment'

draw_barchart(x, y, x_ticks, y_ticks, title)

df_unequal = df.loc[df['Unequal'] == 1, ['Female Tasks', 'Male Maternity', 'Pressure']]
total_unequal = len(df_unequal)

print('% Unequal: {0}'.format(percentage(total_unequal, total)))

df_unequal.head()

female_tasks = percentage(df_unequal['Female Tasks'].value_counts()[1], total_unequal)
male_maternity = percentage(df_unequal['Male Maternity'].value_counts()[0], total_unequal)
unequal_pressure = percentage(df_unequal['Pressure'].value_counts()[1], total_unequal)
print("Female Tasks: {0} | Male Maternity: {1} | Pressure {2}".format(female_tasks, male_maternity, unequal_pressure))

unequal_tasks_percentage = percentage(df_unequal['Female Tasks'].sum(), total_unequal)
unequal_malemat_percentage = percentage(df_unequal['Male Maternity'].sum(), total_unequal)
unequal_pressure_percentage = percentage(df_unequal['Pressure'].sum(), total_unequal)

y = [unequal_tasks_percentage, unequal_malemat_percentage, unequal_pressure_percentage]
x=[0, 1, 2]
x_ticks = df_unequal.columns.values
y_ticks = range(0, 100, 10)
title = 'A Graph to Show Reasons Why Japanese Women Feel the Workplace is Unequal'

draw_barchart(x, y, x_ticks, y_ticks, title)

skilled = df.loc[df['Skilled'] == 1]
total_skilled = len(skilled)

print('% Skilled: {0}'.format(percentage(total_skilled, total)))

skilled.head()

skilled_unequal = df.loc[(df['Skilled'] == 1) & (df['Unequal'] == 1)]
total_skilled_unequal = len(skilled_unequal)

print('% Skilled, Unequal: {0}'.format(percentage(total_skilled_unequal, total_skilled)))

skilled_unequal.head()

skilled_pressure = df.loc[(df['Skilled'] == 1) & (df['Pressure'] == 1)]
total_skilled_pressure = len(skilled_pressure)

print('% Skilled, Pressure: {0}'.format(percentage(total_skilled_pressure, total_skilled)))

skilled_pressure.head()

skilled_disadv = df.loc[(df['Skilled'] == 1) & (df['Disadvantaged'] == 1)]
total_skilled_disadv = len(skilled_disadv)

print('% Skilled, Disadvantaged: {0}'.format(percentage(total_skilled_disadv, total_skilled)))

skilled_disadv.head()

unskilled = df.loc[df['Skilled'] == 0]
total_unskilled = len(unskilled)

print('% Unskilled: {0}'.format(percentage(total_unskilled, total)))

unskilled.head()

unskilled_unequal = df.loc[(df['Skilled'] == 0) & (df['Unequal'] == 1)]
total_unskilled_unequal = len(unskilled_unequal)

print('% Unskilled, Equal: {0}'.format(percentage(total_unskilled_unequal, total_unskilled)))

unskilled_unequal.head()

unskilled_pressure = df.loc[(df['Skilled'] == 0) & (df['Pressure'] == 1)]
total_unskilled_pressure = len(unskilled_pressure)

print('% Unskilled, Pressure: {0}'.format(percentage(total_unskilled_pressure, total_unskilled)))

unskilled_pressure.head()

unskilled_disadv = df.loc[(df['Skilled'] == 0) & (df['Disadvantaged'] == 1)]
total_unskilled_disadv = len(unskilled_disadv)

print('% Unskilled, Disadvantaged: {0}'.format(percentage(total_unskilled_disadv, total_unskilled)))

unskilled_disadv.head()

percentage_skilled_unequal = percentage(total_skilled_unequal, total_skilled)
percentage_unskilled_unequal = percentage(total_unskilled_unequal, total_unskilled)

y = [percentage_skilled_unequal, percentage_unskilled_unequal]
x = [0, 1]
x_ticks = ['Skilled', 'Unskilled']
y_ticks = range(0, 100, 10)
title = 'A Graph to Show the Percentage of Skilled vs Unskilled Women Who Feel Unequal Treatment'

draw_barchart(x, y, x_ticks, y_ticks, title)

managerial = df[df['Managerial'] == 1]
total_managerial = len(managerial)

print('% Managerial: {0}'.format(percentage(total_managerial, total)))

df.head()

female_boss = df[df['Male Boss'] == 0]
total_female_boss = len(male_boss)

y = [percentage(total_managerial, total), percentage(total_female_boss, total)]
x = [0, 1]
x_ticks = ['Managerial Aspirations', 'Female Boss']
y_ticks = range(0, 100, 10)
title = 'A graph to show the difference between women aspiring for managerial positions vs women in managerial positions'

draw_barchart(x, y, x_ticks, y_ticks, title)

quit = df[df['Quit'] == 1]
total_quit = len(quit)

print('% Quit after marriage: {0}'.format(percentage(total_quit, total)))

pressure = df[df['Pressure'] == 1]
total_pressure = len(pressure)

y = [percentage(total_pressure, total), percentage(total_quit, total)]
x = [0, 1]
x_ticks = ['Pressure to quit', 'Want to quit']
y_ticks = range(0, 100, 10)
title = 'A graph to show percentage of Japanese women who want to quit work after marriage vs women who feel a pressure to quit'

draw_barchart(x, y, x_ticks, y_ticks, title)

