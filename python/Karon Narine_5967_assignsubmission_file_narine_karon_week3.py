get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

series1 = pd.Series([1,3,5,np.nan,np.nan,8])

series1

series1[3]

series1[3 :6]

series2 = pd.DataFrame(series1)

series2

s1 , s2 = pd.Series(np.random.randn(6)), pd.Series(np.random.randn(6))

s1

s2

df_1 = pd.DataFrame({'A': series1, 'B':s1, 'C':s2})

df_1

s3 = pd.Series(np.random.randn(8))

df_11 = pd.DataFrame({'A':series1, 'B':s1,'C':s2,'D':s3})

df_11

df2 = pd.DataFrame(np.random.randn(6,4))

df2

df2.tail(2)

df2.columns=('a','b','c','d')

cols = ('a','b','c','d')

df2.columns=cols

df2

dates = pd.date_range(start='2013-02-12 13:45:27', few='W', periods=6)
df2.index= dates

df2

print 'what is a date_range object?\n\n',
dates

df2.loc['2013-02-12 13:45:27']

df2.loc['2013-02-12 13:45:27','c']

df3 = pd.DataFrame(np.random.randn(6,4), index=list('abcdef'),columns=list('ABCD'))

df3

df3.loc[['d','f'],'B':'D']

karon = "id|postedTime|body|None|['twitter_entiteis:urls:url']|['None']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|None|None|None|None|actor:utcOffset|None|None|None|None|None|None|None|None|None|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|None|None|None"
colnames = karon.split('|')

pd.set_option("display.max_columns",None) #this add scroll
df4 = pd.read_csv('../data/twitter_sample.csv', sep='|',names=colnames)
df4.tail(7)

print df4.columns

print df4.drop(  [u'None.5', u'None.6', u'None.7', u'None.8', u'None.9', u'None.10',
       u'None.11', u'None.12', u'None.13'],axis=1).columns

df4.tail(6)

print df4.columns

print df4.drop( [u'None.5', u'None.6', u'None.7', u'None.8', u'None.9', u'None.10',

       u'None.11', u'None.12', u'None.13'],axis=1).columns

df4.tail(6)

df4[df4.twitter_lang=='en'].head()

df5 = df4[["gnip:klout_score","actor:followersCount","actor:friendsCount", "actor:listedCount"]]

df5.head()

df5.dtypes

def floatify(val):
    if val == None or val == 'None':
        return 0.0
    else:
        return float(val)

df5['gnip:klout_score'] = df5['gnip:klout_score'].map(floatify)

df5.dtypes

df5 = df5.astype(float)

df5.dtypes

df5['fol/fr']= df5['gnip:klout_score']/df5['actor:followersCount']

df5

df5.head()

df4.head()

pop_df=df4[df4["actor:followersCount"] >=100]

pop_df

pop_df.dtypes('actor:followersCount')

pop_df.dtypes



pop_df['gnip:klout_score'] = pop_df['gnip:klout_score'].map(floatify)

pop_df.head()

pop_df['Tweet'].dtypes

pop_df.dtypes

pop_df.groupby("twitter_lang").size()

pop_df.groupby("twitter_lang").sum()

pop_df.groupby("twitter_lang").mean()

new_lang = pop_df[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')

new_lang.head(7)

new_lang.dtypes

new_lang.head(2)

new_lang

new_lang.head(2)

new_lang_mean = new_lang.aggregate(np.mean)

new_lang_mean

new_lang_mean.groupby("gnip:klout_score").mean()

new_lang_mean.index

new_lang_mean.head()

new_lang_mean['actor:followersCount'].plot(kind='bar',color='g')

new_lang_mean['gnip:klout_score'].plot(kind='bar',color='r')

plt.scatter(x=new_lang_mean['actor:followersCount'],           y=new_lang_mean['gnip:klout_score'],           alpha=0.5,           s=50,           color='red',           marker='o')

from pandas.plotting import scatter_matrix

scatter_matrix(new_lang_mean, alpha=0.5, figsize=(12,12), diagonal='kde', s=100)

dater = pd.DataFrame(np.random.randn(1000,4), index=pd.date_range('1/1/2000', periods=1000),columns=list('ABCD'))

dater.head()

dater.plot()

dater.hist()

import prettyplotlib

dater.plot()

dater.hist()

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

k = np.random.normal(size=100)

k

import seaborn as sns
sns.set(color_codes=True)

t=sns.distplot(k)

sns.distplot(k)

sns.distplot(k, kde=False, rug=True);

sns.distplot(k, bins=25, kde=False , rug = True)

sns.distplot(k, hist=False, rug=True)

n = np.random.normal(0,1, size=4)
bandwidth = 1.06 * n.std() * n.size ** (-1/5.)
support = np.linspace(-4,4,200)


kernels = []
for n_i in n:
    kernel = stats.norm(n_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support,kernel,color = "r")

sns.rugplot(n,color=".2", linewidth=3);

p = np.random.normal(0,6, size=10)

p

sns.rugplot(p,color = ".3",linewidth=4)

test = np.linspace(-4,4,200)

kernels

n_i

density = np.sum(kernels, axis=0)
density /= integrate.trapz(density,support)
plt.plot(support,density)

support

sns.kdeplot(p, shade=False);

sns.kdeplot(p)

sns.kdeplot(p,bw=.2, label="bw:0.2")

sns.kdeplot(p, bw=.2, label="bw:0.2")
sns.kdeplot(p, bw=2, label="bw:2")
plt.legend();

sns.kdeplot(p,shade=True, cut=0)
sns.rugplot(p);

b = np.random.normal(0,6, size=10)

sns.kdeplot(b,shade=True, cut=2)
sns.rugplot(b);

t = np.random.gamma(6, size=200)

t

sns.distplot(t, kde=True, fit=stats.gamma)

mean, cov = [0,1],[(1,.5),(.5,1)]

cov

meta = np.random.multivariate_normal(mean,cov,200)

meta

ti = pd.DataFrame(meta,columns=["x","y"])

ti

sns.jointplot(x="x",y="y", data = ti)

sns.jointplot(x="x",y="y", data=ti)

x,y = np.random.multivariate_normal(mean,cov,1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x,y=y, kind="hex", color="k");

y

sns.jointplot(x="x",y="y", data=ti, kind="kde");

f,ax = plt.subplots(figsize=(7,7))
sns.kdeplot(ti.x,ti.y, ax=ax)
sns.rugplot(ti.x, color="g", ax=ax)
sns.rugplot(ti.y, vertical=True, ax=ax);

bb=sns.kdeplot(ti.x,ti.y)

bb

f,pp = plt.subplots(figsize=(6,6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(ti.x,ti.y,cmap=cmap, n_levels=60, shade=True);

r = sns.jointplot(x="x",y="y", data=ti, kind="kde", color="m")
r.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
r.ax_joint.collections[0].set_alpha(0)
r.set_axis_labels("$X$","$Y$");

iris = sns.load_dataset("iris")

sns.pairplot(iris);

h = sns.PairGrid(iris)
h

h=sns.PairGrid(iris)
h.map_diag(sns.kdeplot)
h.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord,"regression")))

tips = sns.load_dataset("tips")

tips.tail()

sns.regplot(x="total_bill", y="tip", data=tips);

tipreg = sns.regplot(x="total_bill", y="tip", data=tips);

tipreg;

tips.head()

sns.lmplot(x="total_bill", y="size", data=tips);

sns.lmplot(x="size", y="total_bill", data=tips);

tips.tail()

sns.lmplot(x="total_bill",y="tip", data=tips, x_jitter=.05)
sns.lmplot(x="total_bill", y="tip", data=tips, x_estimator= np.mean);

sns.lmplot(x="size",y="tip", data=tips, x_estimator=np.mean)

anscombe = sns.load_dataset("anscombe")

anscombe.head()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset== 'I'"),
ci=None, scatter_kws={"s":80});

anscombe

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'")
,ci=None, scatter_kws={"s":80});

sns.lmplot(x="x",y="y", data=anscombe.query("dataset=='III'"), 
           robust=True, ci=None, scatter_kws={"s":80});

tips.head()

tips["big_tip"]= (tips.tip/tips.total_bill)> .15

tips.tail()

sns.lmplot(x="total_bill", y="big_tip", data=tips, y_jitter=.03);

sns.lmplot(x="total_bill", y="tip", data=tips,
          lowess=True)

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
          markers=["o","x"], palette="Set1")

sns.lmplot(x="total_bill", y="tip", hue="smoker", col='time', data=tips);

tips.head()

sns.lmplot(x="total_bill", y="tip", hue="smoker", col='time', row="sex", data=tips);

f, ax = plt.subplots(figsize=(5,6))
sns.regplot(x="total_bill",y="tip", data=tips, ax=ax)

sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");

sns.pairplot(tips, x_vars=["total_bill","size"], y_vars=["tip"],
            size=5, aspect=.8, kind="reg");

import numpy as np

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(sum(map(ord,"categorical")))

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.stripplot(x="smoker", y="total_bill", data=tips, jitter=True);

sns.swarmplot(x="smoker",y="total_bill", data=tips);

sns.swarmplot(x="smoker",y="total_bill", hue="sex", data=tips)

sns.swarmplot(x="total_bill",y="day", hue="smoker", data=tips);

sns.boxplot(x="smoker", y="total_bill", hue="sex", data=tips)

sns.violinplot(x="tip",y="day", hue="smoker", data=tips)

sns.barplot(x="sex", y="survived", hue="class", data=titanic);

sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5);



