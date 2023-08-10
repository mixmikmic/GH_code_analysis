import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

df = pd.read_csv("flight_data_cleaned.csv")

df.head()

df.origin.value_counts()

#df.pivot(columns="origin", values="dep_delay")

sns.boxplot(data=np.log(df.pivot(columns="origin", values="dep_delay") + 45) )

df.pivot(columns="origin", values="dep_delay").describe()

avg_dep_delays = df.pivot_table(index="origin", columns="month", values="dep_delay")
sns.heatmap( avg_dep_delays, annot=True )
plt.title("Airport-wise Avg dep_delay (monthly)");

# No. of flights departing various month
count_flight_launches = df.pivot_table(index="origin", columns="month", values="dep_delay", aggfunc=np.size)
sns.heatmap( count_flight_launches )
plt.title("Airport traffic (monthly)");
# So LGA is having least load

sns.heatmap( avg_dep_delays / count_flight_launches );
plt.title("Airport inefficiencies (monthly overview)");

mom_pct_change = ( avg_dep_delays / count_flight_launches).pct_change(axis=1)
mom_pct_change.T.plot();
plt.title("month-on-month Percentage change in inefficiencies");

