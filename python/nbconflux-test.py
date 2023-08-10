import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = sns.load_dataset('planets')

df.head()

df.method.value_counts()

# this cell's input is hidden
fig, ax = plt.subplots(figsize=(8,5))
ax = sns.violinplot(x="orbital_period", y="method",
                    data=df[df.orbital_period < 1000],
                    cut=0, scale="width", palette="Set3",
                    ax=ax)

print("This cell output is hidden.")

