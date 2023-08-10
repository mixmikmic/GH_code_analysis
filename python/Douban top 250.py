import pandas as pd

df = pd.read_csv('group4 - douban/Douban Top 250.csv')

df.head()

df['year'].value_counts()

def cleaning(x):
    if x in ['2004(中国大陆)']:
        return False
    else:
        return True
s_year = df[df['year'].apply(cleaning)]['year'].value_counts()
s_year

from matplotlib import pyplot as plt

plt.plot(s_year.index, s_year.values)

s_year = s_year.sort_index()
plt.plot(s_year.index, s_year.values)
plt.xticks([1, 10, 20, 30, 40, 50, 60,70],['1964','1974','1984','1994','2004','2014','2017'])

s_year = s_year.sort_index()
plt.plot(s_year.index, s_year.values)
plt.xticks([1, 10, 20, 30, 40, 50, 60,70],['1964','1974','1984','1994','2004','2014','2017'])

s_year.index.min()

s_year.index.max()

len(s_year.index)

s_year = s_year.sort_index()
plt.plot(s_year.index, s_year.values)
ticks_index = range(0, 55, 5)
plt.xticks(ticks_index, s_year.index[ticks_index])

plt.figure(figsize=(10, 5))
s_year = s_year.sort_index()
plt.plot(s_year.index, s_year.values)
ticks_index = range(0, 55, 5)
plt.xticks(ticks_index, s_year.index[ticks_index])
plt.title('Top movies by year')
plt.xlabel('Year')
plt.ylabel('# of movies')
plt.annotate('Only a few movies prior 1970', xy=(10, 2), xytext=(10, 8),
            arrowprops=dict(facecolor='black', width=1),
            )
plt.annotate('A sharp increase in the 90s', xy=(28, 10), xytext=(10, 2),
            arrowprops=dict(facecolor='black', width=1),
            )

plt.figure(figsize=(10, 5))
s_year = s_year.sort_index()
plt.bar(s_year.index, s_year.values, color='#75d2e5', edgecolor='#758be5')
ticks_index = range(0, 55, 5)
plt.xticks(ticks_index, s_year.index[ticks_index])
plt.title('Top movies by year')
plt.xlabel('Year')
plt.ylabel('# of movies')
plt.annotate('Only a few movies prior 1970', xy=(10, 2), xytext=(10, 8),
            arrowprops=dict(facecolor='black', width=1),
            )
plt.annotate('A sharp increase in the 90s', xy=(28, 10), xytext=(10, 2),
            arrowprops=dict(facecolor='black', width=1),
            )



