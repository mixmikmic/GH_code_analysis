from jupyter_cms.loader import load_notebook

eda = load_notebook('./data_exploration.ipynb')

df, newspapers = eda.load_data()

slugs_of_interest = [
    'WSJ',
    'USAT',
    'CA_LAT',
    'CA_MN',
    'NY_DN',
    'DC_WP',
    'IL_CST',
    'CO_DP',
    'IL_CT',
    'TX_DMN'
]

import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', 100)

df.head(2)

df['month'] = df['date'].apply(lambda x: x.month)

def print_row(i, row):
    print("#{i}: {title} — {date:%b. %-d} — {fontsize:.2f}pt".format(
        i=i + 1,
        title=" ".join(row.text.split()),
        date=row.date,
        fontsize=row.fontsize))
    
def largest_font_headlines(npdf, paper):
    npdf = npdf[(npdf.bbox_top > npdf.page_height / 2) & (npdf.month >= 6)]
    top = npdf.sort_values(by='fontsize', ascending=False).head(10)
    print(paper)
    for i, (_, row) in enumerate(top.iterrows()):
        print_row(i, row)

# Um, definitely should have a better place for doing this, but on Dec 18th the WSJ PDF I archived was actually
# a different newspaper, somehow. I wonder if it's a Newseum error, but they don't keep their archives up beyond a day

largest_font_headlines(df[(df.slug == 'WSJ') & (df.date != datetime(2017, 12, 18))], 'The Wall Street Journal')

print()

largest_font_headlines(df[df.slug == 'USAT'], 'USA Today')

print()

largest_font_headlines(df[df.slug == 'CA_LAT'], 'Los Angeles Times')

print()

largest_font_headlines(df[df.slug == 'CA_MN'], 'San Jose Mercury News')

print()

largest_font_headlines(df[df.slug == 'NY_DN'], 'New York Daily News')

print()

largest_font_headlines(df[df.slug == 'DC_WP'], 'The Washington Post')

print()

largest_font_headlines(df[df.slug == 'IL_CST'], 'Chicago Sun Times')

print()

largest_font_headlines(df[df.slug == 'CO_DP'], 'The Denver Post')

print()

largest_font_headlines(df[df.slug == 'IL_CT'], 'Chicago Tribune')

print()

largest_font_headlines(df[df.slug == 'TX_DMN'], 'The Dallas Morning News')

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(df.groupby(['slug']).page_height.first())
plt.suptitle("Distribution of page heights (by pixels)")

print('''Heights of known papers:

Broadsheets:
The Washington Post: {}px
The Wall Street Journal: {}px

Tabloids:
The Chicago Sun Times: {}px
The New York Daily News: {}px
'''.format(
    df[df.slug == 'DC_WP'].page_height.mode().iloc[0],
    df[df.slug == 'WSJ'].page_height.mode().iloc[0],
    df[df.slug == 'IL_CST'].page_height.mode().iloc[0],
    df[df.slug == 'NY_DN'].page_height.mode().iloc[0]
))

print('''Aspect ratios of known papers:

Broadsheets:
The Washington Post: {}
The Wall Street Journal: {}

Tabloids:
The Chicago Sun Times: {}
The New York Daily News: {}
'''.format(
    df[df.slug == 'DC_WP'].aspect_ratio.mode().iloc[0],
    df[df.slug == 'WSJ'].aspect_ratio.mode().iloc[0],
    df[df.slug == 'IL_CST'].aspect_ratio.mode().iloc[0],
    df[df.slug == 'NY_DN'].aspect_ratio.mode().iloc[0]
))

from scipy import stats
import numpy as np

def mode(heights):
    return stats.mode(heights).mode[0]

daily_headlines = df.groupby(['date', 'slug']).agg({'fontsize': max, 'page_height': mode, 'aspect_ratio': mode})

daily_headlines.head()

avg_size_by_paper = daily_headlines.reset_index().groupby('slug').agg({'fontsize': np.mean, 'page_height': mode, 'aspect_ratio': mode, 'slug': 'count'}).rename(columns={'slug': 'n'})
avg_size_by_paper.head()

sns.distplot(avg_size_by_paper['n'], kde=False, bins=30)
plt.xlim([0, 250])
plt.suptitle("Distribution of number of days each paper has records in the scrape")

avg_size_by_paper['n'].describe()

avg_size_highly_present = avg_size_by_paper[avg_size_by_paper['n'] > 182]  # more than the median

sns.regplot(avg_size_highly_present.page_height, avg_size_highly_present.fontsize, fit_reg=False)
plt.xlabel("Page height in pixels")
plt.ylabel("Average font point of day's largest headline")
plt.suptitle("Each dot is a newspaper")

sns.regplot(avg_size_highly_present.aspect_ratio, avg_size_highly_present.fontsize, x_jitter=0.01, fit_reg=False)
plt.xlabel("Aspect ratio (width/height)")
plt.ylabel("Average font point of day's largest headline")
plt.suptitle("Each dot is a newspaper")

sns.regplot(avg_size_highly_present.aspect_ratio, avg_size_highly_present.fontsize, x_jitter=0.05, fit_reg=False)

avg_size_highly_present.sort_values(by='fontsize', ascending=False).head(10)

sns.regplot(avg_size_highly_present.aspect_ratio, avg_size_highly_present.page_height, x_jitter=0.01, fit_reg=False)

