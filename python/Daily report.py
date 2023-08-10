get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
start_date = datetime.datetime(2017, 4, 7)
end_date = start_date + datetime.timedelta(days=1)
url = 'http://osmcha.mapbox.com/?date__gte={}&date__lte={}&is_suspect=False&is_whitelisted=All&checked=All&all_reason=True&render_csv=True'.format(start_date.date(), end_date.date())

changesets = pd.read_csv(url)
changesets.head(2)

total_changesets_count = changesets.drop_duplicates('ID').shape[0]
print('Number of changesets: {}'.format(total_changesets_count))

reviewed_changesets_count = changesets[changesets['checked'] == True].drop_duplicates('ID').shape[0]
print('Number of changesets reviewed: {}'.format(reviewed_changesets_count))

print('Percentage changesets reviewed: {}%'.format(round(100.0 * reviewed_changesets_count / total_changesets_count, 2)))

harmful_changesets_count = changesets[changesets['harmful'] == True].drop_duplicates('ID').shape[0]
print('Number of harmful changesets: {}'.format(harmful_changesets_count))

print('Percentage changesets reviewed harmful: {}%'.format(round(100.0 * harmful_changesets_count / reviewed_changesets_count, 2)))

value_counts = changesets['reasons__name'].value_counts()
ax = value_counts.plot.barh()

for i, v in enumerate(value_counts.values):
    ax.text(v + 5, i - 0.18, str(v), fontsize=8)

# List of reasons on osmcha-django from osm-compare
comparators = [
    'Deleted an object having disputed tag',
    'Edited an object having disputed tag',
    'Added invalid highway tag',
    'Added a large building',
    'Edited a major road',
    'Edited a path road',
    'Edited a major lake',
    'New footway created',
    'Edited a place',
    'Major name modification',
    'Deleted a wikidata/wikipedia tag',
    'Feature near Null Island',
    'Feature with Pokename',
    'Dragged highway/waterway',
    'Edited a landmark wikidata/wikipedia',
    'Edited a place wikidata',
    'Edited an osm landmark',
    'Edited an old monument',
    'Added a new place(city/town/country)',
    'New user created a new water feature',
    'Invalid tag combination',
    'Invalid tag modification',
    'Invalid key value combination'
]

flagged_changesets = changesets[changesets['reasons__name'].isin(comparators)]
print('Changesets flagged by osm-compare: {}'.format(flagged_changesets.drop_duplicates('ID').shape[0]))

value_counts = flagged_changesets['reasons__name'].value_counts()
ax = value_counts.plot.barh()

for i, v in enumerate(value_counts.values):
    ax.text(v + 5, i - 0.16, str(v), fontsize=10)

table = []
for comparator in comparators:
    comparator_changesets = changesets[changesets['reasons__name'] == comparator]
    comparator_changesets_reviewed = comparator_changesets[comparator_changesets['checked'] == True]
    comparator_changesets_harmful = comparator_changesets[changesets['harmful'] == True]
    table.append([
        comparator,
        comparator_changesets.shape[0],
        comparator_changesets_reviewed.shape[0],
        comparator_changesets_harmful.shape[0]
    ])

table = sorted(table, key=lambda x: x[1], reverse=True)
# Insert the headers appropriately.
table.insert(0, [' --- '] * 4)
table.insert(0, ['comparator', 'flagged', 'flagged_and_reviewed', 'flagged_and_harmful'])

for row in table:
    print(' | '.join([str(item) for item in row]))

