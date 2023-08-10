import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

df = pd.read_csv('timeuse.csv')

import dateutil.parser

def string_time(str):
    return dateutil.parser.parse(str)

df['FormattedStartTime'] = df['StartTime'].apply(string_time)

type(df['FormattedStartTime'][0])

df.head()

df.columns

df[df['Country'] == 'Finland'].sort_values('Eating', ascending = False).head()

df['FormattedStartTime'].describe()

b = df[df['Country'] == 'Belgium'][['Eating', 'FormattedStartTime']]
b = b.set_index('FormattedStartTime')
print(b[:25])
# b.plot(b['FormattedStartTime'], b['Eating'])

c_df = pd.read_csv('hours_coordinates.csv')

temp = []
for x in c_df.sort_values('hitemp', ascending=False)['country']:
    temp.append(x)
temp

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
num = 0
for country in temp:
    num += 1
    b = df[df['Country'] == country][['Eating', 'FormattedStartTime']]
    b = b.set_index('FormattedStartTime')
    fig, ax = plt.subplots(figsize=(8, 5))
    b.plot(linewidth = 1, ax = ax, legend = False)
    ax.set_ylim([0, 100])
    ax.set_yticklabels('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tick_params(
        which = 'major',
        top = 'off',
        bottom = 'off',
        left = 'off',
        right = 'off',
        labeltop = 'off',
        labelbottom = 'off'
        )
    plt.tick_params(
        which = 'minor',
        top = 'off',
        bottom = 'off',
        left = 'off',
        right = 'off',
        labeltop = 'off',
        labelbottom = 'off'
        )
    ax.set_xlabel(country)
    plt.savefig(str(num)+country+'.pdf', transparent = True, bbox_inches='tight')

c_df['percent_temp'] = (c_df['hitemp']-43.9) / 25.4 * 100 # 69.3-43.9

c_df['percent_shading'] = (c_df['hitemp']-35) / 34.5 * 100 # 69.3-43.9

c_df.sort_values('hitemp', ascending=False)[['country', 'hitemp', 'percent_temp', 'percent_shading']]



