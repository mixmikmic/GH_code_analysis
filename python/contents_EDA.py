from collections import Counter
from itertools import chain, combinations
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.set(font_scale=2)

filename = 'data/contents_py_000000000000'

data = []
for line in open(filename, 'r'):
    data.append(json.loads(line))

data[0]

data[1]

df = pd.read_json(json.dumps(data), orient='records')

df.head()

df.info()

# Remove rows without content
df = df[df['content'].notnull()]

def extract_package_names(df):
    '''
    Input (df): Pandas dataframe with code content column named 'content'
    Output (df): Dataframe with added column of lists - of packages imported for each code content
    '''
    # Match package names following 'from' or following 'import', but not those following 'from ... import'
    df['packages'] = df['content'].str.findall(r"from ([\w-]+) import [\w*-]+|import ([\w-]+)")
    # Flatten packages into one list
    df['packages'] = df['packages'].apply(lambda x: list(set([i for i in chain.from_iterable(x) if i != ''])))
    # NaN for rows with no package imports 
    df['packages'] = df['packages'].map(lambda x: np.NaN if len(x)==0 else x)
    return df

df = extract_package_names(df)

df.head()

df.info()

# Remove code contents with no package imports
df = df[df['packages'].notnull()]

df.info()



# Count packages
pkg_counter = Counter(chain(*(df['packages'].values)))
# pkg_counter = sorted(pkg_counter.items(), key=lambda x: -x[1])

# Plot top 30
def plot_top(counter, n=30, title='Packages'):
    '''
    Input (df_series): Pandas series of list of packages imported in code content
    Input (n): top n packages by count to plot
    Output: bar plot by count
    '''
    plot_df = pd.DataFrame(counter.most_common(n))
    plot_df.columns = ['package', 'count']
    g = sns.factorplot(x='package', y='count', data=plot_df, kind='bar', aspect=2, size=6)
    g.set_xticklabels(rotation=90)
    plt.title('Count of Top {} {}'.format(n, title))
    plt.show()

plot_top(pkg_counter, 30, 'Packages')

pkg_edges = []

f = open('pkg_edges.tsv', 'w')

for linked_pkgs in df['packages']:
    for pkg_edge in combinations(linked_pkgs, 2):
        # Append edge to list
        pkg_edges.append(tuple(sorted(pkg_edge)))
        # Write edge to file
        f.write("{}\t{}\n".format(pkg_edge[0], pkg_edge[1]))

f.close()

# Count package pairs (edges)
pkg_edge_counter = Counter(pkg_edges)

pkg_edge_counter.most_common(10)

plot_top(pkg_edge_counter, 30, 'Edges')

df_from = pd.DataFrame(list(df['pkg_from'].apply(lambda x: tuple(set(x))).values))

df_from.head()

df_import = pd.DataFrame(list(df['pkg_import'].apply(lambda x: tuple(set(x))).values))

df_import.head()

