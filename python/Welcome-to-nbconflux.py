get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = sns.load_dataset("anscombe")
df.groupby('dataset').describe()['y']

sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1});

