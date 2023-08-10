get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

df = pd.read_csv("New_York_City_Leading_Causes_of_Death.csv")

df

df.columns

df['Count'].plot.box()

df.groupby('Year')['Count'].sum().sort_values(ascending=False)

df.groupby('Year')['Count'].sum().mean()

fig, ax = plt.subplots(figsize=(9, 6))
df.groupby('Year')['Count'].sum().plot.barh()
mean = df.groupby('Year')['Count'].sum().mean()
ax.plot([mean, mean], [0, 12], c='blue', linestyle="-", linewidth=0.5)
ax.annotate(s="Mean of death registered, 190,629.6", xy=(120000,0), color='Blue')

df.groupby('Sex')['Count'].sum()

df.groupby(['Year', 'Sex'])['Count'].sum()

fig, ax = plt.subplots(figsize=(9, 7))
df.groupby(['Year', 'Sex'])['Count'].sum().plot(color=['darkred', 'blue'],kind='bar', title="deaths by gender over time")
ax.set_ylabel('total deaths')
ax.set_ylabel('Sex')
ax.set_ylim((0,110000))

df.groupby(['Cause of Death', 'Sex'])['Count'].sum().sort_values(ascending=False).head(6)

df.groupby(['Cause of Death', 'Sex'])['Count'].sum().sort_values(ascending=False)

#Disease of heart is the number 1
df.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(1)

df.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(9, 7))
df.groupby('Cause of Death')['Count'].sum().sort_values(ascending=True).plot.barh()
ax.set_xlim((0,400000))

df.groupby('Ethnicity')['Count'].sum().sort_values(ascending=False)

df.groupby('Ethnicity')['Count'].sum().plot.barh(color=['Black', 'Black', 'Black', 'darkred'])

fig, ax = plt.subplots(figsize=(9, 7))
only_whites = df[df['Ethnicity'] == 'NON-HISPANIC WHITE']
only_whites.groupby('Cause of Death')['Count'].sum().sort_values(ascending=True).plot.barh(color='blue')

fig, ax = plt.subplots(figsize=(9, 7))
only_blacks = df[df['Ethnicity'] == 'NON-HISPANIC BLACK']
only_blacks.groupby('Cause of Death')['Count'].sum().sort_values(ascending=True).plot.barh(color='black')

fig, ax = plt.subplots(figsize=(9, 7))
only_hispanic = df[df['Ethnicity'] == 'HISPANIC']
only_hispanic.groupby('Cause of Death')['Count'].sum().sort_values(ascending=True).plot.barh(color='brown')

only_asians = df[df['Ethnicity'] == 'ASIAN & PACIFIC ISLANDER']

fig, ax = plt.subplots(figsize=(9, 7))
only_asians.groupby('Cause of Death')['Count'].sum().sort_values(ascending=True).plot.barh(color='green')

df['Year'].value_counts()

df1 = df[df['Year'] == 2011]
df2 = df[df['Year'] == 2010]
df3 = df[df['Year'] == 2009]
df4 = df[df['Year'] == 2008]
df5 = df[df['Year'] == 2007]

fig, ax = plt.subplots(figsize=(7, 9))
df1.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(10).plot.bar()
ax.set_title("2011")
ax.set_ylim((0,90000))
plt.savefig("heart1.pdf")

fig, ax = plt.subplots(figsize=(7, 9))
df2.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(10).plot.bar()
ax.set_title("2010")
ax.set_ylim((0,90000))
plt.savefig("heart2.pdf")

fig, ax = plt.subplots(figsize=(7, 9))
df3.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(10).plot.bar()
ax.set_title("2009")
ax.set_ylim((0,90000))
plt.savefig("heart3.pdf")

fig, ax = plt.subplots(figsize=(7, 9))
df4.groupby('Cause of Death')['Count'].sum().sort_values(ascending=False).head(10).plot.bar()
ax.set_title("2008")
ax.set_ylim((0,90000))
plt.savefig("heart4.pdf")





