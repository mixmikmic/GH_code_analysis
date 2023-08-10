get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np1
import pandas as pd1
import matplotlib.pyplot as plt1 

n1 = pd.Series([1,3,5,np1.nan,6,8])
n1

n1[4]

n2, n3 = pd.Series(np.random.randn(6)), pd.Series(np.random.randn(6))

dr_1 = pd.DataFrame({'A': n1, 'B': n2, 'C': n3})

dr_1

n4 = pd.Series(np.random.randn(8))  # whoaaaaaa this Series has extra entries!

df1 = pd.DataFrame({'A': n1, 'B': n2, 'C': n3, 'D': n4})

df1 

df2 = pd.DataFrame(np.random.randn(6,4))

df2             # can only have one 'pretty' output per cell (if it's the last command)

print df2       # otherwise, can print arb number of results w/o pretty format
print df1       # (uncomment both of these print statements)

cols = ['col1', 'col2', 'col3', 'col4']

# assign columns attribute (names) 
df2.columns = cols

# create an index:
#  generate a sequence of dates with pandas' data_range() method,
#  then assign the index attribute
dates = pd.date_range(start='2017-01-01 13:45:27', freq='W', periods=6)
df2.index = dates

df2

gnacs_x = "id|postedTime|body|None|['twitter_entiteis:urls:url']|['None']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|None|None|None|None|actor:utcOffset|None|None|None|None|None|None|None|None|None|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|None|None|None"
colnames = gnacs_x.split('|')

# prevent the automatic compression of wide dataframes (add scroll bar)
pd.set_option("display.max_columns", None)

# get some data, inspect
df1 = pd.read_csv('../data/twitter_sample.csv', sep='|', names=colnames)

df1

# inspect those rows with twitter-classified lang 'en' (scroll the right to see)
df1[df1.twitter_lang == 'en'].head()

df2 = df1[["gnip:klout_score","actor:followersCount", "actor:friendsCount", "actor:listedCount"]]

df2

df2.dtypes  

def floatify(val):
    if val == None or val == 'None':
        return 0.0
    else:
        return float(val)

df2['gnip:klout_score'] = df2['gnip:klout_score'].map(floatify)
df2.dtypes

df2 = df2.astype(float)

df2.dtypes

df2['New Column:fol/fr'] = df2['gnip:klout_score'] / df2['actor:followersCount']

df2.head()

pop_df = df1[df1["actor:followersCount"] >= 100]
pop_df.body.head()

lang_gb = pop_df[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')

lang_gb.head(2)  

lang_gb_mean = lang_gb.aggregate(np.mean)  

lang_gb_mean.head()

lang_gb_mean.index

lang_gb_mean['actor:friendsCount'].plot(kind='bar', color='b')

plt.scatter(x=lang_gb_mean['actor:followersCount'],            y=lang_gb_mean['actor:friendsCount'],            alpha=0.5,            s=50,            color='blue',            marker='o')

from pandas.plotting import scatter_matrix

scatter_matrix(lang_gb_mean, alpha=0.5, figsize=(12,12), diagonal='kde', s=100)

df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2000', periods=1000), columns=list('ABCD'))
df = df.cumsum()
df.head()

df.plot()
df.hist()

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

x = np.random.normal(size=100)
sns.distplot(x);

sns.distplot(x, kde=False, rug=True);

sns.distplot(x, bins=1000, kde=False, rug=True);

sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend();

sns.kdeplot(x, shade=True);

x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma);

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

sns.jointplot(x="x", y="y", data=df);

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k");

sns.jointplot(x="x", y="y", data=df, kind="kde");

iris = sns.load_dataset("iris")
sns.pairplot(iris);

g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Reds_d", n_levels=10);

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

sns.lmplot(x="size", y="tip", data=tips);



anscombe = sns.load_dataset("anscombe")

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80});

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

sns.swarmplot(x="size", y="total_bill", data=tips);

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips);

sns.boxplot(x="day", y="total_bill", hue="time", data=tips);

