import os
import sys

# Set path to root directory so we can access core, tasks, data, etc.
os.chdir('..')
sys.path.insert(0,  os.path.realpath('core/ext_libs'))
from core.ext_libs.vislab.datasets import ava

# Get dataframe
df = ava.get_ava_df()
len(df)

# Get one image by id
df[df.index == '1048']

# Print some stats, show many images would be in 0-1 split depending 
# on how we handle the 'neutral' items
# Note: 5 used in the paper
print df.rating_mean.min()
print df.rating_mean.max()
print df.rating_mean.mean()
print df.rating_mean.std()
print df.rating_std.mean()
print df.rating_std.std()

print len(df[df.rating_mean < 5])
print len(df[df.rating_mean > 5])
print len(df[df.rating_mean < 4])
print len(df[df.rating_mean > 6])
print len(df[df.rating_mean < 4.25])
print len(df[df.rating_mean > 5.75])
print len(df[df.rating_mean < 4.33])
print len(df[df.rating_mean > 5.67])

# Add labels
def label_f(x, bin_edges):
    for i in range(len(bin_edges)):
        if x < bin_edges[i]:
            return int(i-1)
        
bin_edges = [0, 5.0, 10.0]
    
# df_sub.dropna(inplace=True)
df['label'] = df.rating_mean.apply(label_f, args=(bin_edges, ))

print len(df[df.label == 0])
print len(df[df.label == 1])

# Save image_id and labels
id2label = {}
for id, row in df.iterrows():
    id2label[id] = row.label

# Work with subset of relevant columns
df_sub = df[['rating_mean', 'rating_std', 'ratings']]

# Filter to those where the rating_std is <= mean + 1std
df_sub = df_sub[df_sub.rating_std < (df_sub.rating_std.mean() + df_sub.rating_std.std())]

# Print some things

print df_sub.rating_mean.min()
print df_sub.rating_mean.max()
print df_sub.rating_mean.mean()
print df_sub.rating_mean.std()
print df_sub.rating_std.mean()
print df_sub.rating_std.std()

# Add labels
def label_f(x, bin_edges):
    for i in range(len(bin_edges)):
        if x < bin_edges[i]:
            return int(i-1)
        
bin_edges = [0, 5.0, 10.0]
    
# df_sub.dropna(inplace=True)
df_sub['label'] = df_sub.rating_mean.apply(label_f, args=(bin_edges, ))

print len(df_sub[df_sub.label == 0])
print len(df_sub[df_sub.label == 1])

