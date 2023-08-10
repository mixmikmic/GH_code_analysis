import pandas as pd
import numpy as np

gnacs_x = "id|postedTime|body|None|['twitter_entiteis:urls:url']|['None']|['actor:languages_list-items']|gnip:language:value|twitter_lang|[u'geo:coordinates_list-items']|geo:type|None|None|None|None|actor:utcOffset|None|None|None|None|None|None|None|None|None|actor:displayName|actor:preferredUsername|actor:id|gnip:klout_score|actor:followersCount|actor:friendsCount|actor:listedCount|actor:statusesCount|Tweet|None|None|None"
colnames = gnacs_x.split('|')

#del df1['None']

cols = list(df1.columns)
cols


df1[df1.twitter_lang == 'ru'].tail(2)

df2 = df1[["gnip:klout_score","actor:followersCount", "actor:friendsCount", "actor:listedCount"]]

df2.tail(2)

df2.dtypes 

def floatify(val):
    if val == None or val == 'None':
        return 0.0
    else:
        return float(val)

df2['gnip:klout_score'] = df2['gnip:klout_score'].map(floatify)

# check again
df2.dtypes

df2 = df2.astype(float)

df2.dtypes


df2['fol/fr'] = df2['gnip:klout_score'] / df2['actor:followersCount']

df2.head()

df1.head()

pop_df = df1[df1["actor:followersCount"] >= 100]
pop_df.tail(3)

lang_gb = pop_df[['twitter_lang',             'gnip:klout_score',             'actor:followersCount',             'actor:friendsCount',             'actor:statusesCount']].groupby('twitter_lang')
lang_gb.tail()

lang_gb_mean = lang_gb.aggregate(np.mean)  

lang_gb_mean.head()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

lang_gb_mean['actor:followersCount'].plot(kind='bar', color='g')

plt.scatter(x=lang_gb_mean['actor:followersCount'],            y=lang_gb_mean['actor:friendsCount'],            alpha=0.5,            s=350,            color='red',            marker='o')

from pandas.plotting import scatter_matrix

scatter_matrix(lang_gb_mean, alpha=0.5, figsize=(12,12), diagonal='kde', s=100)

import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(1000, 4), index=pd.date_range('1/1/2000', periods=1000), columns=list('ABCD'))
df = df.cumsum()
df.tail()

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

sns.distplot(x, bins=150, kde=False, rug=True);

sns.distplot(x, hist=False, rug=True);

x = np.random.normal(0, 1, size=50)
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

x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma);

mean, cov = [0, 5], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
df.head()

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

import seaborn as sns
sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips);

sns.lmplot(x="total_bill", y="tip", data=tips);

sns.lmplot(x="size", y="tip", data=tips);

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05);

sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean);

anscombe = sns.load_dataset('anscombe')
anscombe

anscombe = sns.load_dataset("anscombe")
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           ci=None, scatter_kws={"s": 80});

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           order=2, ci=None, scatter_kws={"s": 80});

tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           y_jitter=.03);

sns.lmplot(x="total_bill", y="big_tip", data=tips,
           logistic=True, y_jitter=.03);

sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
              scatter_kws={"s": 80});

sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
              scatter_kws={"s": 80});

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1");

sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);

f, ax = plt.subplots(figsize=(5, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           col_wrap=2, size=3);

sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", size=5, aspect=.8, kind="reg");

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.stripplot(x="day", y="total_bill", data=tips);

sns.swarmplot(x="day", y="total_bill", data=tips);

sns.swarmplot(x="size", y="total_bill", data=tips);

sns.swarmplot(x="total_bill", y="day", hue="time", data=tips);

sns.boxplot(x="day", y="total_bill", hue="time", data=tips);

tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False);

sns.violinplot(x="total_bill", y="day", hue="time", data=tips);

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

sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5);

g = sns.PairGrid(tips,
                 x_vars=["smoker", "time", "sex"],
                 y_vars=["total_bill", "tip"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel");

sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm");

