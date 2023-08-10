from os.path import join 

import pandas as pd
import numpy as np

url = 'http://casas.wsu.edu/datasets/twor.2009.zip'
zipfile = url.split('/')[-1]
dirname = '.'.join(zipfile.split('.')[:2])
filename = join(dirname, 'data')

print '     url: {}'.format(url)
print ' zipfile: {}'.format(zipfile)
print ' dirname: {}'.format(dirname)
print 'filename: {}'.format(filename)

column_headings = ('date', 'time', 'sensor', 'value', 'annotation', 'state')

df = pd.read_csv(
    filename, 
    delim_whitespace=True,  # Note, the file is delimited by both space and tab characters
    names=column_headings
)

df.head() 

df.dtypes

df.ix[df.date.str.startswith('22009'), 'date'] = '2009-02-03'

df['datetime'] = pd.to_datetime(df[['date', 'time']].apply(lambda row: ' '.join(row), axis=1))

df.head()

df = df[['datetime', 'sensor', 'value', 'annotation', 'state']]

df.set_index('datetime', inplace=True)

df.head()

df.sensor.unique()

df.annotation.unique()

df.state.unique()

df.value.unique()

categorical_inds = df.sensor.str.match(r"^[^A]")

df_categorical = df.loc[categorical_inds][['sensor', 'value']]

df_categorical.head()

df_categorical.loc[:, 'sensor_value'] = df_categorical[['sensor', 'value']].apply(
    lambda row: '{}_{}'.format(*row).lower(), 
    axis=1
)

df_categorical.head()

df_categorical_exploded = pd.get_dummies(df_categorical.sensor_value)

df_categorical_exploded.head()

df_categorical_exploded.values

numeric_inds = df.sensor.str.startswith("A")

df_numeric = df.loc[numeric_inds][['sensor', 'value']]

df_numeric.head()

f_inds = df_numeric.value.str.endswith('F')

df_numeric.loc[f_inds, 'value'] = df_numeric.loc[f_inds, 'value'].str[:-1]

df_numeric.loc[f_inds]

df_numeric.value = df_numeric.value.map(float)

unique_keys = df_numeric.sensor.unique()

unique_keys

df_numeric = pd.merge(df_numeric[['value']], pd.get_dummies(df_numeric.sensor), left_index=True, right_index=True)

df_numeric.head()

for key in unique_keys:
    df_numeric[key] *= df_numeric.value

df_numeric = df_numeric[unique_keys]

# Print a larger sample of the data frame
df_numeric

df_categorical_exploded.head()

df_numeric.head()

df_joined = pd.merge(
    df_categorical_exploded, 
    df_numeric, 
    left_index=True, 
    right_index=True,
    how='outer'
)

df_joined.head()

annotation_inds = pd.notnull(df.annotation)

df_annotation = df.loc[annotation_inds][['annotation', 'state']]

# There are some duplicated indices. Remove with
df_annotation = df_annotation.groupby(level=0).first()

df_annotation.head()

for annotation, group in df_annotation.groupby('annotation'): 
    counts = group.state.value_counts()
    
    if counts.begin == counts.end: 
        print '             {}: equal counts ({}, {})'.format(
            annotation, 
            counts.begin, 
            counts.end
        )
        
    else:
        print ' *** WARNING {}: inconsistent annotation counts with {} begin and {} end'.format(
            annotation, 
            counts.begin, 
            counts.end
        )

df_annotation.loc[df_annotation.annotation == 'R1_Work']

def filter_annotations(anns):
    left = iter(anns.index[:-1])
    right = iter(anns.index[1:])

    inds = []
    for ii, (ll, rr) in enumerate(zip(left, right)): 
        try:
            l = anns.ix[ll]
            r = anns.ix[rr]

            if l.state == 'begin' and r.state == 'end': 
                inds.extend([ll, rr])
                
        except ValueError:
            print ii 
            print l
            print
            print r
            print
            print 
            
            asdf

    return anns.loc[inds, :]
        

dfs = []
for annotation, group in df_annotation.groupby('annotation'): 
    print '{:>30} - {}'.format(annotation, group.size)
    dfs.append(filter_annotations(group))

df_annotation_exploded = pd.get_dummies(df_annotation.annotation)

df_annotation_exploded

paired = pd.concat(dfs)

left = paired.index[:-1:2]
right = paired.index[1::2]

print df_annotation_exploded.mean()
for ll, rr in zip(left, right): 
    l = paired.ix[ll]
    r = paired.ix[rr]
    
    assert l.annotation == r.annotation
    
    annotation = l.annotation
    begin = l.name
    end = r.name
    
    # Another advantage of using datetime index: can slice with time ranges
    df_annotation_exploded.loc[begin:end, annotation] = 1

df_annotation_exploded.head()

dataset = pd.merge(
    df_joined, 
    df_annotation_exploded, 
    left_index=True, 
    right_index=True, 
    how='outer'
)

data_cols = df_joined.columns
annotation_cols = df_annotation_exploded.columns

dataset[data_cols] = dataset[data_cols].fillna(0)
dataset[annotation_cols] = dataset[annotation_cols].ffill()

dataset.head()

dataset[data_cols].head()

dataset[annotation_cols].head()

















