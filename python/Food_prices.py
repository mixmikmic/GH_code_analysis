import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import dateutil.parser
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

df = pd.read_excel('table97_2014.xlsx')
#df = df.transpose()

df_continents = pd.read_csv('continents_countries.csv', sep=';', usecols=('Country (en)', 'Continent'))
df_continents.index = df_continents['Country (en)']
df_continents = df_continents.drop('Country (en)', axis=1)

df_merge = df.merge(df_continents, how='left', left_index=True, right_index=True)
#df_merge[df_merge['Continent'] != df_merge['Continent']] -> be sure that all got a continent
df_merge.head()

df_distribution = pd.read_excel('distribution_clean.xls', usecols=['Country', 'Gini index'])
print(df_distribution.columns)
df_distribution.head()


df_sample = df.sample(n=5, weights='Food')
df_sample = df_sample.append(df.loc['USA'])
df_sample = df_sample.append(df.loc['Italy'])
df_sample = df_sample.append(df.loc['Nigeria'])
df_sample = df_sample.append(df.loc['Singapore'])
df_sample = df_sample.append(df.loc['United Kingdom'])
df_sample = df_sample.drop_duplicates()
df_sample = df_sample.sort_values('Food', ascending=False)

ax = df_sample['Food'].plot(kind='barh', figsize=(20, 6))
ax.set_title('Percentage of expenses for food (selected countries)')
plt.savefig('Percentage_income.pdf')

#df = df.merge(on='')
#df_distribution.index = df_distribution['Country']
#df_distribution = df_distribution.drop('Country', axis=1)

df_distribution = df_distribution[df_distribution['Gini index'] != '..']
df_distribution.head()

#df_merge = df.merge(df_distribution, left_index=True, right_index=True)

df_captions = df.sample(n=5, weights='Food')
df_captions = df_captions.append(df.loc['USA'])
df_captions = df_captions.append(df.loc['Italy'])
df_captions = df_captions.append(df.loc['Nigeria'])
df_captions = df_captions.append(df.loc['Singapore'])
df_captions = df_captions.append(df.loc['Switzerland'])
df_captions = df_captions.append(df.loc['Australia'])
df_captions = df_captions.append(df.loc['United Kingdom'])
df_captions = df_captions.drop_duplicates()
df_captions = df_captions.sort_values('Food', ascending=False)

for name, country in df_captions.iterrows():
    print(name, country['Food'])

#df_merge.plot(kind='scatter', x='Consumer expenditures', y='Food')

fig, ax = plt.subplots(figsize=(10,10))

colors = ['#1f78b4','#b2df8a','#a6cee3','#33a02c','#fb9a99','#e31a1c','#fdbf6f']
i = 0
for continent, group in df_merge.groupby('Continent'):
    group.plot(kind='scatter', x='Consumer expenditures', s=100, y='Food', ax=ax, c=colors[i], alpha=.9, label=continent) # cmap=plt.cm.coolwarm
    i += 1

for name, row in df_captions.iterrows():
    x_gap = 800
    ax.text(row['Consumer expenditures']+x_gap, row['Food']-.4, name )

plt.savefig('Food_prices_continents.pdf')

set(df_merge['Continent'].values)

def get_continent_index(continent_name):
    continent_names = ['Africa',
 'Asia',
 'Australia',
 'Central America',
 'Europe',
 'North America',
 'South America']
    try:
        return continent_names.index(continent_name)
    except:
        print("Not in list:", continent_name)
get_continent_index('Australia')

df_merge['Continent_id'] = df_merge['Continent'].apply(get_continent_index)

color_indexes = list(df_merge.sort_values(by='Food', ascending=False)['Continent_id'])
color_list = [colors[i] for i in color_indexes]
color_list[:10]

import matplotlib as mpl

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=0, vmax=6)
#cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

sorted_df = df_merge.sort_values(by='Food', ascending=False)

ax = sorted_df['Food'].plot(kind='barh', color=color_list, figsize=(20, 18))
ax.set_title('Percentage of expenses for food (all monitored countries)')

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

rectangles = [i for i in ax.get_children() if isinstance(i, matplotlib.patches.Rectangle)]
for rect in rectangles:
    width = rect.get_width()
    ax.text(width + 1.8, rect.get_y() - height/3.,
            "{0:.2f}%".format(width),
            ha='center', va='bottom')


lines = []
continent_names = ['Africa',
 'Asia',
 'Australia',
 'Central America',
 'Europe',
 'North America',
 'South America']
for i in range(0, 7):
    line, = plt.plot([], c=colors[i], label=continent_names[i])
    lines.append(line)
plt.legend(handles=lines)
plt.savefig('barh.pdf')

children = ax.get_children()
for child in children:
    if isinstance(child, matplotlib.patches.Rectangle):
        print('Rectangle')

for index, row in sorted_df.iterrows():
    print("{0:.2f}%".format(row['Food']))

ax = df['Food'].sort_values(ascending=False).plot(kind='barh', figsize=(20, 18))
ax.set_title('Percentage of expenses for food (all monitored countries)')
plt.savefig('Percentage_income_full.pdf')









dfp = pd.read_excel('food_prices.xlsx')

# does not work
#def parse_date(str_date):
#    return dateutil.parser.parse(str(str_date))
#df['Year'] = df['Year'].apply(parse_date)

dfp.head()

dfp.index = dfp['Year']

# df = df.drop('Year', axis=1)

df_recent = dfp[dfp.Year >= 2000]
df_recent.head()
#dfp.columns

ax = plt.subplot()

df_recent['Eggs'].plot(ax=ax, figsize=(20, 8))
df_recent['Pork'].plot()
df_recent['Fish and seafood'].plot()
df_recent['Nonalcoholic beverages'].plot()

ax.legend()
ax.set_title('In the last 15 years, eggs were much more volatile than beverage, pork or fish and seafood')
ax.set_ylabel('Price index')
plt.savefig('Eggs_volatility.pdf')

df.loc[1975] - df.loc[2015]

(df.loc[2015] - df.loc[1975]).plot(kind='barh')

(df.loc[2015] - df.loc[2000]).plot(kind='barh')

