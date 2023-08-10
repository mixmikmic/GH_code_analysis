get_ipython().run_line_magic('matplotlib', 'inline')
# import naming conventions 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# (further reading on mpl imports: http://bit.ly/197aGoq )

m1 = pd.Series([np.nan,4,8,16,24,17,14])
m1

# by default (without specifying them explicitly), the index label is just an int
m1[6]

# create a couple more Series
m2, m3 = pd.Series(np.random.randn(7)), pd.Series(np.random.randn(7))

# combine multiple Series into a DataFrame with column labels
mydf = pd.DataFrame({'W': m1, 'X': m2, 'Y': m3})

mydf

# when Series are different lengths, DataFrame fills in gaps with NaN
m4 = pd.Series(np.random.randn(9))  # whoaaaaaa this Series has extra entries!

my_df = pd.DataFrame({'W': m1, 'X': m2, 'Y': m3, 'Z': m4})

my_df 

# create a DataFrame from numpy array
mydf1 = pd.DataFrame(np.random.randn(8,5))

mydf1             # can only have one 'pretty' output per cell (if it's the last command)

print mydf1       # otherwise, can print arb number of results w/o pretty format
print mydf       # (uncomment both of these print statements)

# recall current dataframe 
mydf1.head(4)

cs = ['x', 'y', 'z', 'v', 'w']

# assign columns attribute (names) 
mydf1.columns = cs

# create an index:
#  generate a sequence of dates with pandas' data_range() method,
#  then assign the index attribute
dts = pd.date_range(start='2013-05-24 16:14:14', freq='W', periods=8)
mydf1.index = dts

mydf1

# an aside: inspecting the dates object...
print 'what is a date_range object?\n\n', dts

# select a row by index label by using .loc 
mydf1.loc['2013-07-07 16:14:14']

# select a single element
mydf1.loc['2013-05-26 16:14:14','z']

# new dataframe with random numbers
df_1 = pd.DataFrame(np.random.randn(7,7), index=list('rstlnea'),columns=list('WXYZABC'))

df_1

# address three separate rows, and a range of three columns
df_1.loc[['l','e','n'],'X':'Z']

g_x = "id|postedTime|body|NA|['twitter_entiteis:urls:url']|['NA']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|NA|NA|NA|NA|actor:utcOffset|NA|NA|NA|NA|NA|NA|NA|NA|NA|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|NA|NA|NA"
cnames = g_x.split('|')

# prevent the automatic compression of wide dataframes (add scroll bar)
pd.set_option("display.max_columns", None)

# get some data, inspect
d1 = pd.read_csv('../data/twitter_sample.csv', sep='|', names=cnames)

d1.tail(10)

# n.b.: this is an *in-place* delete -- unusual for a pandas structure
#del d1['NA'] This way didn't work, so I did it below

# The command below is how the docs suggest carrying this out (creating a new df). 
#   But, it doesn't seem to work -- possibly due to multiple cols with same name. Oh well. 
new_d1 = d1.drop([u'NA', u"['NA']", u'NA.1', u'NA.2', u'NA.3',u'NA.4',u'NA.5', u'NA.6', u'NA.7', u'NA.8',                  u'NA.9',u'NA.10', u'NA.11', u'NA.12', u'NA.13',u'NA.14', u'NA.15', u'NA.16'], axis=1)  # return new df

# have a peek
new_d1.tail(5)
new_d1.columns

# inspect those rows with twitter-classified lang 'en' (scroll the right to see)
new_d1[new_d1.twitter_lang == 'it'].head()

# the colons in the column name below won't allow dot-access to the column, so we can quote them and still filter.
new_d1[new_d1["gnip:language:value"] == 'it'].head()  

# create new dataframe from numerical columns
d2 = new_d1[["gnip:klout_score","actor:followersCount", "actor:friendsCount", "actor:listedCount","actor:statusesCount"]]

d2.head()

# because I happen to know the answer, let's check data types of the columns...
d2.dtypes  

# convert ints / strings to floats, give up on anything else (call it 0.0)
def makefloaty(arg):
    if arg == None or arg == 'None':
        return 0.0
    else:
        return float(arg)

# assigning to an existing column overwrites that column 
d2['gnip:klout_score'] = d2['gnip:klout_score'].map(makefloaty)

# check again
d2.dtypes

# use all floats just for fun. 
#  this only works if the elements can all be converted to floats (e.g. ints or something python can handle) 
d2 = d2.astype(float)

d2.dtypes

# look at some activity ratios - add col to df
d2['fol/fr'] = d2['gnip:klout_score'] / d2['actor:followersCount']
d2['fr/flw'] = d2['actor:friendsCount'] / d2['actor:followersCount']

d2.head()

# can also use the built-in describe() method to get quick descriptive stats on the dataframe
#d2.describe() -- I did this in a separate line so I could keep the head output pretty....

d2.describe()

# back to bigger df, without 'None' cols
new_d1.head()

# subset df, create new df with only 'popular' accounts -- those matching the filter condition given
trendy_df = new_d1[new_d1["actor:followersCount"] >= 150]

# fix the klout scores again
trendy_df['gnip:klout_score'] = trendy_df['gnip:klout_score'].map(makefloaty)

# in case you need to remind yourself of the dataframe
trendy_df.head()

# use GroupBy methods for stats on each group:
trendy_df.groupby("twitter_lang").size()      # number of elements per group
trendy_df.groupby("twitter_lang").sum()       # sum of elements in each group (obviously doesn't make sense for some cols) 
trendy_df.groupby("twitter_lang").mean()      # algebraic mean of elements per group

# though this looks like a normal dataframe, the DataFrameGroupBy object has a heirarchical index
#  this means it may not act as you might expect.
lingo_gb = trendy_df[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')


# note the new index 'twitter_lang' -- in this case, .head(n) returns <= n elements for each index
lingo_gb.head(1)  

# see that they type is DataFrameGroupBy object
lingo_gb

# to get a DataFrame object that responds more like I'm used to, create a new one using the 
#   aggregate method, which results in a single-index DataFrame
lingo_gb_mean = lingo_gb.aggregate(np.mean)  

lingo_gb_mean.head()

# verify the single index
lingo_gb_mean.index

# .plot() is a pandas wrapper for matplotlib's plt.plot() 
lingo_gb_mean['actor:followersCount'].plot(kind='bar', color='b')

# more base matplotlib 
plt.scatter(x=lingo_gb_mean['actor:followersCount'],            y=lingo_gb_mean['actor:friendsCount'],            alpha=0.5,            s=50,            color='blue',            marker='o')

# now read the docs and copypasta a neat-looking plot
from pandas.plotting import scatter_matrix

scatter_matrix(lingo_gb_mean, alpha=0.5, figsize=(12,12), diagonal='hist', s=100)
#I changed the diagonals to histograms...just wanted to see how that worked.

# make up some data with large-scale patterns and a datetime index
d3 = pd.DataFrame(np.random.randn(1000, 5), index=pd.date_range('1/1/2010', periods=1000), columns=list('XYZVW'))
d3 = d3.cumsum()
d3.head()

d3.plot()
d3.hist()

#!pip install prettyplotlib
import prettyplotlib

d3.plot()
d3.hist()

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=100)
sns.distplot(x);

sns.rugplot(x) #decided to try to do a rugplot, it would be good if you're short on space, I guess.

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k");

iris = sns.load_dataset("iris")

g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(sum(map(ord, "categorical")))

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
               split=True, inner="stick", palette="Set3");

