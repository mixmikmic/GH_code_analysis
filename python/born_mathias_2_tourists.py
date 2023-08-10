import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Read in the csv.
df = pd.read_csv("data/tourism_bern_UTF8.csv", sep=";", header=0)
df.head(3)

# Grab the new header for later use.  
header = df['Herkunftsland']

# Change the orientation of the dataframe. 
df = df.transpose()
df.head(3)

# Correction to the header
df = df.ix[1:]
df.columns = header
df.head(3)

# Convert first row to a timeserie. 
df.index = pd.to_datetime(df.index)
df.head(5)

plt.style.use('fivethirtyeight')

# Let's check how the total number overnight stays evolved over time.
df.sum(axis=1).plot()

# Let's check how many overnight stays from Swiss people there are in total.
df['Schweiz'].sum()

# How many guests there are in total for every year? 
df.resample('A').sum().sum(axis=1)

# What countries besides Switzerland are the most guest coming from? 
country_dict = {}
for i in header:
    country_dict[i] = int(df[i].sum())
df_total = pd.DataFrame(country_dict, index=[0])
df_total = df_total.transpose()
df_total.columns = ['total']
best_guest = df_total.sort_values('total', ascending=False).head(15)
best_guest

# Making a graph of the most important countries for the hotels in Bern.

df_total.loc[['Deutschland', 'Vereinigte Staaten von Amerika', 'Frankreich', 'Italien', 'Vereinigtes Königreich']].plot(kind='barh')

# Making a chart of the most important countries for the hotels in Bern. Attempt II. 

fig, ax = plt.subplots(figsize=(20,8))
best_guest.plot(kind='barh', x=best_guest.index, y='total', legend=None, ax=ax)
ax.set_title('Ohne die Deutschen bleiben die Berner Hotels halb leer')
ax.set_xlabel('Logiernächte 2005 -- 2015')
ax.set_ylabel('')
ax.grid(linestyle=':', linewidth='0.4', color='red')
plt.savefig("bestguest.pdf", transparent=True)

# Let's take this one over to Inkscape to do some polishing...

# Here's another attempt.

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(df[['Deutschland', 'Vereinigte Staaten von Amerika', 'Frankreich', 'Italien', 'Vereinigtes Königreich']])
ax.set_title('Bern\'s best guest')
ax.set_ylabel('Number of nights')
ax.set_xlabel('')
plt.savefig("bestguest.pdf", transparent=True)

# Let's have a look at the people from Germany.

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Deutschland'])
# ax.axvline(x='2015-01-15', ymin=0, ymax=.5, label='Aufwertung CH-Franken')
ax.set_title('Wo bleiben die Deutschen?')
ax.set_ylabel('Hotel nights per month')

# Is there a correlation with the development with the Swiss franc?
# Source of the data: Swiss National Bank, https://data.snb.ch/de/topics/ziredev#!/cube/devkum

df_money = pd.read_csv('snb.csv')
df_euro = df_money[df_money['D1'] == 'EUR1']
df_euro.index = pd.to_datetime(df_euro['Date'])
del df_euro['Date']
df_euro.head()

df_euro.head()

df_deutschland

ax = df[0:10].plot(y='Deutschland')
ax

df_euro[0:10].plot(y='Value', color='darkblue', kind='bar', ax=ax)

ax

# Let's add the money conversion rate to the graphic.

fig, ax1 = plt.subplots(figsize=(10,8), sharex=True)
ax1.plot(df['Deutschland'], color='crimson')
#ax1.axvline(x='2015-01-15', ymin=0, ymax=1)
ax1.set_title('Die Deutschen meiden Bern. Ein Grund: der starke Franken')
ax1.set_ylabel('Übernachtungen (Monat)', color="crimson")

ax2 = ax1.twinx()

ax2.plot(df_euro['Value'], color='darkblue')
ax2.set_ylabel('Franken pro Euro', color='darkblue')
plt.savefig('europroblem.pdf', tranparent=True)

# Let's add the money conversion rate to the graphic.

fig, ax1 = plt.subplots(figsize=(10,7), sharex=True)
ax1.plot(df['Deutschland'], color='crimson')
#ax1.axvline(x='2015-01-15', ymin=0, ymax=1)
ax1.set_title('Die Deutschen meiden Bern. Ein Grund: der starke Franken')
ax1.set_ylabel('Übernachtungen (Monat)', color="crimson")
plt.savefig('europroblem_1.pdf')

# Maybe seperated is better. Let's work with this one in Inkscape. 

# Let's add the money conversion rate to the graph.

fig, ax2 = plt.subplots(figsize=(10,2), sharex=True)

ax2.plot(df_euro['Value'], color='darkblue')
ax2.set_ylabel('Franken pro Euro', color='darkblue')
plt.savefig('europroblem_2.pdf', tranparent=True)

# Let's take this one over to Inkscape to add to the last one for some polishing. 

