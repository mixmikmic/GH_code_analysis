import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

employee_data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print('Total entries: ' + str(len(employee_data)))
employee_data.head(5)

list(employee_data.columns.values)

employee_data.EnvironmentSatisfaction.head(5)

employee_data.Attrition.replace(['Yes', 'No'], [1, 0], inplace=True)
employee_data.Attrition.head(5)

employee_data.Attrition.value_counts()

fig = employee_data[['Attrition', 'EnvironmentSatisfaction']].groupby('EnvironmentSatisfaction').mean().plot()
plt.title('% Attrition vs Environmental Satisfaction (ES)')
plt.ylabel('% Attrition')
plt.show()

employee_data[['Attrition', 'EnvironmentSatisfaction']].groupby('EnvironmentSatisfaction').mean()

employee_data.EnvironmentSatisfaction.value_counts().sort_index().plot(kind='bar')
plt.title('Employees per ES Value')
plt.ylabel('Employees')
plt.xlabel('Environmental Satisfaction')
plt.show()

employee_data.EnvironmentSatisfaction.value_counts().sort_index()

before = np.zeros(1470) #The control group: Attrition is 237 of 1470
before[:237] = 1

after = np.zeros(1470)  #Best case scenario: Attrition is 209 of 1470
after[:209] = 1

stats.mannwhitneyu(before, after)

before = np.zeros(571) #The control group: Attrition only for ES=1 or 2: 115 of 571
before[:115] = 1

after = np.zeros(571)  #Best case test group: Attrition reduced by 28: 87 of 571
after[:87] = 1

stats.mannwhitneyu(before, after)

delta = 150  #Number of employees moving from ES=1 to ES=2

plt.bar([1, 2, 3, 4], [284-delta, 287+delta, 453, 446])
plt.ylabel('Employees')
plt.xlabel('ES Rating')
plt.title('Hypothetical ES Distribution After Intervention')
plt.show()

# First, let's look at the null hypothesis. Are the before/after samples from the same population?

# ES=1 now includes only 134 employees, ES=2 is 437. Total = 571
# Attrition for ES=1 is 134 * 0.25 = 34
# Attrition for ES=2 is 437 * 0.15 = 66   Total Attrition: 100

before = np.zeros(571) #Control group: 115 of 571
before[:115] = 1

after = np.zeros(571)  #Test group: 100 of 571
after[:100] = 1

stats.mannwhitneyu(before, after)

# And now looking at the company as a whole
# N.B. We expect to be less confident when looking at the larger group.

before = np.zeros(1470) #Control group: Before intervention: 237 of 1470
before[:237] = 1

after = np.zeros(1470)  #Test group: Attrition reduced by 15: 222 of 1470
after[:222] = 1

stats.mannwhitneyu(before, after)



