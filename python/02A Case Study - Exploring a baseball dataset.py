#from scipy.stats import futil
from ggplot import *
import pandas as pd
import numpy as np

get_ipython().magic('matplotlib inline')

df = pd.read_csv("baseball-pitches-clean.csv")
df = df[['pitch_time', 'inning', 'pitcher_name', 'hitter_name', 'pitch_type', 
         'px', 'pz', 'pitch_name', 'start_speed', 'end_speed', 'type_confidence']]
df.head()

ggplot(df, aes(x='px', y='pz')) + geom_point()

ggplot(aes(x='start_speed', y='end_speed'), data=df) + geom_point()

ggplot(aes(x='start_speed'), data = df) + geom_histogram()
# had to change "rows" to "index" in ggplot/stats/stat_bin.py, 
# on line 126-127 manually for this to work

df_pitchname = df.groupby("pitch_name")
for name, frame in df_pitchname:
    print ggplot(aes(x='start_speed'), data=frame) + geom_histogram() + ggtitle("Distribution of " + str(name))

ggplot(aes(x='start_speed'), data=df) + geom_histogram() + facet_wrap('pitch_name')

from IPython.display import YouTubeVideo
YouTubeVideo("ikLlRT2j7EQ")

ggplot(aes(x='pitch_type'), data=df) + geom_bar()

ggplot(df,aes(x = 'start_speed')) + geom_histogram() + facet_grid('pitch_type')

#not working
#ggplot(aes(x='start_speed'),data=df)+geom_histogram()+facet_grid('pitch_type','pitch_type')#,scales="free")

ggplot(df, aes(x='start_speed')) +    geom_density()

ggplot(df, aes(x='start_speed', color='pitch_type' )) + geom_density()



