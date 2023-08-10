get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('matplotlib', 'inline')
# import naming conventions 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# (further reading on mpl imports: http://bit.ly/197aGoq )

series1 = pd.Series([1,3,5,np.nan,7,9])
series1

# by default (without specifying them explicitly), the index label is just an int
series1[5]

# create a couple more Series
s2, s3 = pd.Series(np.random.randn(6)), pd.Series(np.random.randn(6))

s3

# combine multiple Series into a DataFrame with column labels
df_1 = pd.DataFrame({'A': s1, 'B': s2, 'C': s3})

df_1

# when Series are different lengths, DataFrame fills in gaps with NaN
s4 = pd.Series(np.random.randn(8))  # whoaaaaaa this Series has extra entries!

df1 = pd.DataFrame({'A': s1, 'B': s2, 'C': s3, 'D': s4})

df1 

# create a DataFrame from numpy array
df2 = pd.DataFrame(np.random.randn(6,4))

df2             # can only have one 'pretty' output per cell (if it's the last command)

#print df2       # otherwise, can print arb number of results w/o pretty format
#print df1       # (uncomment both of these print statements)

# recall current dataframe 
df2.head(2)

dates = pd.date_range(start='2013-11-24 13:45:27', freq='D', periods=6)

dates

cols = ['a', 'b', 'c', 'd']

# assign columns attribute (names) 
df2.columns = cols

# create an index:
#  generate a sequence of dates with pandas' data_range() method,
#  then assign the index attribute
dates = pd.date_range(start='2013-11-24 13:45:27', freq='W', periods=6)
df2.index = dates

df2

# an aside: inspecting the dates object...
print 'what is a date_range object?\n\n', dates

# select a row by index label by using .loc 
df2.loc['2013-12-01 13:45:27']

# select a single element
df2.loc['2013-12-22 13:45:27','c']

# new dataframe with random numbers
df1 = pd.DataFrame(np.random.randn(6,4), index=list('abcdef'),columns=list('ABCD'))

df1

# address two separate rows, and a range of three columns
df1.loc[['d','f'],'A':'C']

gnacs_x = "id|postedTime|body|None|['twitter_entiteis:urls:url']|['None']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|None|None|None|None|actor:utcOffset|None|None|None|None|None|None|None|None|None|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|None|None|None"
colnames = gnacs_x.split('|')

dataframe = pd.read_csv('./data/twitter_sample.csv', sep='|', names=colnames)

dataframe.head()

# prevent the automatic compression of wide dataframes (add scroll bar)
pd.set_option("display.max_columns", None)

# get some data, inspect
df1 = pd.read_csv('./data/twitter_sample.csv', sep='|', names=colnames)

df1.tail()

df1.columns

# n.b.: this is an *in-place* delete -- unusual for a pandas structure
#del df1['None'] 
df2 = df1.drop( [u'None.5', u'None.6', u'None.7', u'None.8', u'None.9', u'None.10', u'None.11', u'None.12', u'None.13'],axis=1) 

# The command below is how the docs suggest carrying this out (creating a new df). 
#   But, it doesn't seem to work -- possibly due to multiple cols with same name. Oh well. 
#new_df = df1.drop('None', axis=1)  # return new df

df2.head(3)

# have a peek
df2.tail(6)

# inspect those rows with twitter-classified lang 'en' (scroll the right to see)
df1[df1.twitter_lang == 'fr'].head(3)

# the colons in the column name below won't allow dot-access to the column, so we can quote them and still filter.
#df1[df1["gnip:language:value"] == 'en'].head()  

# create new dataframe from numerical columns
df2 = df1[["gnip:klout_score","actor:followersCount", "actor:friendsCount", "actor:listedCount"]]

df2.head(10)

# because I happen to know the answer, let's check data types of the columns...
df2.dtypes  

# convert ints / strings to floats, give up on anything else (call it 0.0)
def floatify(val):
    if val == None or val == 'None':
        return 0.0
    else:
        return float(val)

# assigning to an existing column overwrites that column 
df2['gnip:klout_score'] = df2['gnip:klout_score'].map(floatify)

# check again
df2.dtypes

# use all floats just for fun. 
#  this only works if the elements can all be converted to floats (e.g. ints or something python can handle) 
df2 = df2.astype(float)

df2.dtypes

df2['actor:followersCount'] = df2['actor:followersCount'].map(float)

df2.dtypes



# look at some activity ratios - add col to df
df2['fol/fr'] = df2['gnip:klout_score'] / df2['actor:followersCount']

df2.head()

# can also use the built-in describe() method to get quick descriptive stats on the dataframe
df2.describe()

# back to bigger df, without 'None' cols
df1.head()


pop_df = df1[df1["actor:followersCount"] <100 ]

# fix the klout scores again
#pop_df['gnip:klout_score'] = pop_df['gnip:klout_score'].map(floatify)

# in case you need to remind yourself of the dataframe
#pop_df.head()

# use GroupBy methods for stats on each group:
#pop_df.groupby("twitter_lang").size()      # number of elements per group
#pop_df.groupby("twitter_lang").sum()       # sum of elements in each group (obviously doesn't make sense for some cols) 
#pop_df.groupby("twitter_lang").mean()      # algebraic mean of elements per group

pop_df.groupby("actor:followersCount").size().head()

pop_df.groupby("gnip:language:value").mean()

pop_df.groupby("gnip:language:value").sum()

# though this looks like a normal dataframe, the DataFrameGroupBy object has a heirarchical index
#  this means it may not act as you might expect.
lang_gb = pop_df[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')


# note the new index 'twitter_lang' -- in this case, .head(n) returns <= n elements for each index
lang_gb.head(2)  

# see that they type is DataFrameGroupBy object
#lang_gb

pop_df.head()

# to get a DataFrame object that responds more like I'm used to, create a new one using the 
#   aggregate method, which results in a single-index DataFrame
lang_gb_median = lang_gb.aggregate(np.median)  

lang_gb_median.head()

# verify the single index
#lang_gb_mean.index

# .plot() is a pandas wrapper for matplotlib's plt.plot() 
lang_gb_median['actor:followersCount'].plot(kind='bar', color='r')
lang_gb_median['actor:friendsCount'].plot(kind='bar',color='r')
lang_gb_median.plot(x='actor:friendsCount' , 
                    y='actor:followersCount',kind='scatter',color='r', marker='o')

# more base matplotlib 
plt.scatter(x=lang_gb_median['actor:followersCount'],            y=lang_gb_median['actor:statusesCount'],            alpha=0.5,            s=50,            color='red',            marker='o')


# now read the docs and copypasta a neat-looking plot
from pandas.plotting import scatter_matrix

scatter_matrix(lang_gb_median, alpha=1, figsize=(10,10), diagonal='hist', s=100)

# make up some data with large-scale patterns and a datetime index
df = pd.DataFrame(np.arange(1,5), index=pd.date_range('1/1/2000', periods=4), columns=list('T'))

df.head()

df1 = df.cumsum()
df1.head()
df1.append(pd.DataFrame([1] , index=['2000-01-05'] , columns=list('T')))
 

df1.plot()

plt.hist(df1.values)

import prettyplotlib

df.plot()
df.hist()

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=10)
sns.distplot(x);

sns.distplot(x, kde=False, rug=True);

sns.distplot(x, bins=20, kde=False, rug=True);

sns.distplot(x, hist=False, rug=True);

x = np.random.normal(0, 1, size=30)
bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
support = np.linspace(-4, 4, 200)

kernels = []
for x_i in x:

    kernel = stats.norm(x_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support, kernel, color="b")

sns.rugplot(x, color=".008", linewidth=8);

density = np.sum(kernels, axis=0)
density /= integrate.trapz(density, support)
plt.plot(support, density);

sns.kdeplot(x, shade=True);

sns.kdeplot(x)
sns.kdeplot(x, bw=.8, label="bw: 0.8")
sns.kdeplot(x, bw=5, label="bw: 5")
plt.legend();

sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x);

x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma);

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

df

sns.jointplot(x="x", y="y", data=df);

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="b");

sns.jointplot(x="x", y="y", data=df, kind="kde");

f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax);

f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);

g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");

iris = sns.load_dataset("iris")
sns.pairplot(iris);

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

sns.regplot(x="total_bill", y="tip", data=tips);

sns.lmplot(x="total_bill", y="tip", data=tips);

sns.lmplot(x="size", y="tip", data=tips);

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);

sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);

anscombe = sns.load_dataset("anscombe")

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 100});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80});

tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           y_jitter=.03);

sns.lmplot(x="total_bill", y="big_tip", data=tips,
           logistic=True, y_jitter=.03);

sns.lmplot(x="total_bill", y="tip", data=tips,
           lowess=True);

sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
              scatter_kws={"s": 80});

sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
              scatter_kws={"s": 80});

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

sns.stripplot(x="day", y="total_bill", data=tips);

sns.stripplot(x="day", y="total_bill", data=tips, jitter=True);

sns.swarmplot(x="day", y="total_bill", data=tips);

sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips);

sns.swarmplot(x="size", y="total_bill", data=tips);

sns.swarmplot(x="total_bill", y="day", hue="time", data=tips);

sns.boxplot(x="day", y="total_bill", hue="time", data=tips);

tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False);

sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5);

