get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

gnacs_x = "id|postedTime|body|None|['twitter_entiteis:urls:url']|['None']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|None|None|None|None|actor:utcOffset|None|None|None|None|None|None|None|None|None|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|None|None|None"
colnames = gnacs_x.split('|')

pd.set_option("display.max_columns", None)

dataframe1 = pd.read_csv('../data/twitter_sample.csv', sep='|', names=colnames)

dataframe1.head(5)

del dataframe1['None']

dataframe1.head(5)

dataframe1[dataframe1.twitter_lang == 'ru'].head()

dataframe2 = dataframe1[["gnip:klout_score","actor:followersCount", "actor:friendsCount", "actor:listedCount"]]

dataframe2.head()

def floatify(val):
    if val == None or val == 'None':
        return 0.0
    else:
        return float(val)

dataframe2['gnip:klout_score'] = dataframe2['gnip:klout_score'].map(floatify)
dataframe2.dtypes

dataframe2 = dataframe2.astype(float)

dataframe2.dtypes

dataframe2['follower to friend ratio'] = dataframe2['actor:followersCount'] / dataframe2['actor:friendsCount']

dataframe2.head()

dataframe1.head()

popular = dataframe1[dataframe1["actor:friendsCount"] >= 100]

popular['gnip:klout_score'] = popular['gnip:klout_score'].map(floatify)

popular.groupby("twitter_lang").size()

lang_gb = popular[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')


lang_gb.head(3)  

lang_gb_avg = lang_gb.aggregate(np.mean)  

lang_gb_avg.tail()

lang_gb_avg['actor:friendsCount'].plot(kind='bar', color='r')

plt.scatter(x=lang_gb_avg['actor:followersCount'],            y=lang_gb_avg['actor:friendsCount'],            alpha=0.5,            s=50,            color='red',            marker='o')

from pandas.plotting import scatter_matrix

scatter_matrix(lang_gb_avg, alpha=0.5, figsize=(12,12), diagonal='kde', s=100)

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

x = np.random.normal(size=200)
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
    plt.plot(support, kernel, color="r")

sns.rugplot(x, color=".2", linewidth=3);

density = np.sum(kernels, axis=0)
density /= integrate.trapz(density, support)
plt.plot(support, density);

sns.kdeplot(x, shade=True);

sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend();

sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x);

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
           order=2, ci=None, scatter_kws={"s": 80});

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

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");

sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips);

f, ax = plt.subplots(figsize=(5, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           col_wrap=2, size=3);

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           aspect=.5);

sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             size=5, aspect=.8, kind="reg");

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", size=5, aspect=.8, kind="reg");

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

sns.violinplot(x="total_bill", y="day", hue="time", data=tips);

sns.violinplot(x="total_bill", y="day", hue="time", data=tips,
               bw=.1, scale="count", scale_hue=False);

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True);

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
               split=True, inner="stick", palette="Set3");

sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5);

sns.barplot(x="sex", y="survived", hue="class", data=titanic);

sns.countplot(x="deck", data=titanic, palette="Greens_d");

sns.countplot(y="deck", hue="class", data=titanic, palette="Greens_d");

sns.pointplot(x="sex", y="survived", hue="class", data=titanic);

sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"]);

sns.boxplot(data=iris, orient="h");

sns.violinplot(x=iris.species, y=iris.sepal_length);

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="deck", data=titanic, color="c");

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips);

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar");

sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm");

sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5);

g = sns.PairGrid(tips,
                 x_vars=["smoker", "time", "sex"],
                 y_vars=["total_bill", "tip"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel");

