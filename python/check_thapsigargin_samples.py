# of two replicate featurecounts runs, one has an extra column (or another has a missing one)

import pandas as pd
import os

wd = '/home/bay001/projects/parp13_ago2_20171015/permanent_data/chang_newdata/'

df1 = pd.read_table(os.path.join(wd, 'featureCounts/counts.Thapsigargin.multimap.txt'), comment='#')
df2 = pd.read_table(os.path.join(wd, 'featureCounts_newData/counts.Thapsigargin.multimap.txt'), comment='#')

df1.head()

df2.head()

df1.shape

df2.shape



