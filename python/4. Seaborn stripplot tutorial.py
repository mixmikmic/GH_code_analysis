import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# load tips dataset

tips_data = sns.load_dataset("tips")
tips_data.head()

# Draw a single horizontal strip plot:

sns.set_style("whitegrid")
tips_data = sns.load_dataset("tips")
sns.stripplot(x = tips_data["total_bill"],size = 4,color = "green")

sns.stripplot(x = 'day',y ='total_bill',data = tips_data,size =7,order = ['Fri','Sat','Sun','Thur'])


# use of jitter

sns.stripplot(x = 'day',y ='total_bill',data = tips_data,size =7,order = ['Fri','Sat','Sun','Thur'],jitter = True)

# Draw horizontal strips:
sns.stripplot(x="total_bill", y="day", data=tips_data,jitter = True)

# Draw outlines around the points:
sns.stripplot(x="total_bill", y="day", data=tips_data,jitter = True,linewidth = 1)

# Nest the strips within a second categorical variable:

sns.stripplot(x="day", y="total_bill", data=tips_data,jitter = True,hue = "sex")
plt.legend()


# Draw each level of the hue variable at different locations on the major categorical axis:
sns.stripplot(x="day", y="total_bill", data=tips_data,jitter = True,hue = "sex",dodge = True)
plt.legend()

# Control strip order by passing an explicit order:

sns.stripplot(x="time", y="tip", data=tips_data,order = ["Dinner","Lunch"],size = 8)

sns.boxplot(x="tip", y="day", data=tips_data, whis=np.inf)
sns.stripplot(x="tip", y="day", data=tips_data,jitter=True, color=".3")

sns.stripplot(x="day", y="total_bill", data=tips_data, jitter=True)
sns.violinplot(x="day", y="total_bill", data=tips_data,inner=None, color=".8")



