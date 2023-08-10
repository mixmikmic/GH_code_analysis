import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

df = pd.read_csv("qi_episodes.csv")

df.date = pd.to_datetime(df.date)

from IPython.display import display

win_ratio = df.did_alan_win.value_counts(normalize=True)
display(win_ratio)

win_ratio.plot(kind="bar")
plt.xlabel("Did Alan win?")
plt.xticks([0, 1], ["No", "Yes"], rotation=0)
plt.ylabel("Percentage")
plt.show()

# aggregate by series
ts = df.groupby("series")["did_alan_win"].mean() * 100
# plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ts.plot(ax=ax)
ax.set_title("Alan's Win Percentage Over Time")
ax.set_ylabel("% of victories")
ax.set_xticks(range(len(ts.index)))
ax.set_xticklabels(ts.index)
plt.show()

