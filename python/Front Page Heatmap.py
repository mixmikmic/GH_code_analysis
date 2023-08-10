import pandas as pd

df = pd.read_sql_table('frontpage_texts', 'postgres:///frontpages')

df.head()

len(df.groupby(['page_width', 'page_height']).indices)

df['page_width_round'] = df['page_width'].apply(int)
df['page_height_round'] = df['page_height'].apply(int)

len(df.groupby(['page_width_round', 'page_height_round']).indices)

df['page_width_round_10'] = df['page_width'].apply(lambda w: int(w/10)*10)
df['page_height_round_10'] = df['page_height'].apply(lambda w: int(w/10)*10)

print('''Number of unique dimensions: {}

Top dimensions:
{}'''.format(
    len(df.groupby(['page_width_round_10', 'page_height_round_10']).slug.nunique()),
    df.groupby(['page_width_round_10', 'page_height_round_10']).slug.nunique().sort_values(ascending=False)[:10]
))

newspapers = pd.read_sql_table('newspapers', 'postgres:///frontpages')

WIDTH = 790
HEIGHT = 1580

df_at_size = df[(df.page_width_round_10 == WIDTH) & (df.page_height_round_10 == HEIGHT)]

print('Number of days for which we data for each newspaper')
pd.merge(newspapers, df_at_size.groupby('slug').date.nunique().reset_index(), on='slug').sort_values('date', ascending=False)

one_paper = df_at_size[(df_at_size.slug=='NY_RDC') & (df_at_size.date == df_at_size.date.max())]

print('''The Rochester Democrat and Chronicle has {} entries in the database across {} days.

On the latest day, it has {} text fields.
'''.format(
    df_at_size[df_at_size.slug == 'NY_RDC'].shape[0],
    df_at_size[df_at_size.slug == 'NY_RDC'].date.nunique(),
    one_paper.shape[0]
))

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

plt.figure(figsize=(WIDTH/200, HEIGHT/200))
currentAxis = plt.gca()

for i, row in one_paper.iterrows():
    left = row.bbox_left / row.page_width
    right = row.bbox_right / row.page_width
    top = row.bbox_top / row.page_height
    bottom = row.bbox_bottom / row.page_height
    
    currentAxis.add_patch(Rectangle((left, bottom), right-left, top-bottom,alpha=0.5))
    
plt.suptitle('Layout of detected text boxes on a single front page')
plt.show()

import numpy as np

def make_intensity_grid(paper, height=HEIGHT, width=WIDTH, verbose=False):
    intensity_grid = np.zeros((height, width))

    for i, row in paper.iterrows():
        left = int(row.bbox_left)
        right = int(row.bbox_right)
        top = int(row.bbox_top)
        bottom = int(row.bbox_bottom)

        if np.count_nonzero(intensity_grid[bottom:top, left:right]) > 0:
            if verbose:
                print('Warning: overlapping bounding box with', bottom, top, left, right)
        intensity_grid[bottom:top, left:right] = row.avg_character_area
    
    return intensity_grid

def plot_intensity(intensity, title, scale=100):
    height, width = intensity.shape
    fig = plt.figure(figsize=(height/scale, width/scale))
    ax = plt.gca()

    cmap = plt.get_cmap('YlOrRd')
    cmap.set_under(color='white')

    fig.suptitle(title)
    plt.imshow(intensity, cmap=cmap, extent=[0, width, 0, height], origin='lower', vmin=0.1)
    plt.close()
    return fig
    
intensity_grid = make_intensity_grid(one_paper)
plot_intensity(intensity_grid, 'Intensity map of a front page')

intensities = []
for i, ((date, slug), paper) in enumerate(df_at_size.groupby(['date', 'slug'])):
    if i % 50 == 0:
        print('.', end='')
    intensities.append(make_intensity_grid(paper))

avg_intensity = sum(intensities) / len(intensities)
plot_intensity(avg_intensity, 'Average intensity of all {} x {} newspapers'.format(HEIGHT, WIDTH))

df['aspect_ratio'] = df['page_width_round_10'] / df['page_height_round_10']

print('''Out of {} newspapers, there are {} unique aspect ratios.

The top ones are:
{}'''.format(
    df.slug.nunique(),
    df.groupby('slug').aspect_ratio.first().nunique(),
    df.groupby('slug').aspect_ratio.first().value_counts().head(5)
))

import math

df['aspect_ratio'] = np.round(df['page_width_round_10'] / df['page_height_round_10'], decimals=1) 

print('''This time, there are {} unique aspect ratios.

Top ones:
{}'''.format(
    df.groupby('slug').aspect_ratio.first().nunique(),
    df.groupby('slug').aspect_ratio.first().value_counts()
))

smallest_width = df[df.aspect_ratio == 0.5].page_width_round_10.min()
smallest_height = df[df.aspect_ratio == 0.5].page_height_round_10.min()
print('''The easiest way would be to scale down to the smallest dimensions.

{} x {}'''.format(
    smallest_width,
    smallest_height
))

from scipy.misc import imresize

intensities = []
for i, ((date, slug), paper) in enumerate(df[df.aspect_ratio == 0.5].groupby(['date', 'slug'])):
    if i % 50 == 0:
        print('.', end='')
    intensities.append(imresize(make_intensity_grid(paper), (smallest_height, smallest_width)))

count = len(intensities)
avg_intensity = sum([x / count for x in intensities])

plot_intensity(avg_intensity, 'Average front-page of {} newspapers'.format(len(intensities)))

newspapers[newspapers.slug == 'NY_NYT'].head()

def newspaper_for_slug(slug):
    return newspapers[newspapers.slug == slug].title.iloc[0]

def slug_for_newspaper(title):
    return newspapers[newspapers.title == title].slug.iloc[0]

def avg_frontpage_for(newspaper_title='', random=False, paper=df):
    if newspaper_title:
        slug = slug_for_newspaper(newspaper_title)
        if slug not in paper.slug.unique():
            return 'No data'
    elif random:
        slug = paper.sample(1).slug.iloc[0]
        newspaper_title = newspaper_for_slug(slug)
    else:
        raise ArgumentError('Need newspaper_title or random=True')
    
    newspaper = paper[paper.slug == slug]
    width = newspaper.iloc[0].page_width_round
    height = newspaper.iloc[0].page_height_round

    intensities = []
    for i, ((date, slug), paper) in enumerate(newspaper.groupby(['date', 'slug'])):
        intensities.append(make_intensity_grid(paper, height=height, width=width))

    avg_intensity = sum([x / len(intensities) for x in intensities])

    return plot_intensity(avg_intensity, 'Average front-page of {} ({} days)'.format(newspaper_title, newspaper.date.nunique()))

avg_frontpage_for('The Denver Post')

avg_frontpage_for('The Washington Post')

avg_frontpage_for(random=True)

avg_frontpage_for(random=True)

df[df.slug == slug_for_newspaper('Marietta Daily Journal')].text.value_counts().head()

text_counts = df.groupby(['slug']).text.value_counts()

duplicate_text = text_counts[text_counts > 1].reset_index(name='count').drop('count', axis=1)

print('Detected {} rows of duplicate text'.format(duplicate_text.shape[0]))

from collections import defaultdict

duplicate_text_dict = defaultdict(set)
_ = duplicate_text.apply(lambda row: duplicate_text_dict[row.slug].add(row.text), axis=1)

df_clean = df[df.apply(lambda row: row.text not in duplicate_text_dict[row.slug], axis=1)]

avg_frontpage_for('The Hamilton Spectator', paper=df_clean)

avg_frontpage_for('The Washington Post', paper=df_clean)

avg_frontpage_for(random=True, paper=df_clean)

intensities = []
for i, ((date, slug), paper) in enumerate(df_clean[df_clean.aspect_ratio == 0.5].groupby(['date', 'slug'])):
    if i % 50 == 0:
        print('.', end='')
    intensities.append(imresize(make_intensity_grid(paper), (smallest_height, smallest_width)))

count = len(intensities)
avg_intensity = sum([x / count for x in intensities])

plot_intensity(avg_intensity, 'Average front-page of newspapers')

