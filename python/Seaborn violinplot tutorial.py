import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

tips_data = sns.load_dataset("tips")

tips_data.head(7)

sns.violinplot(x = tips_data['total_bill'])


# Draw a vertical violinplot grouped by a categorical variable:
sns.violinplot(x = 'day',y ='total_bill',data = tips_data)

# Draw a violinplot with nested grouping by two categorical variables:
sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted')
plt.legend()


# Draw split violins to compare the across the hue variable:
sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',split = True)
plt.legend()

# Control violin order by passing an explicit order:
sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',split = True,order = ['Fri','Sat','Sun'])

# Scale the violin width by the number of observations in each bin:

sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',split = True,               scale = 'count' )

# Draw the quartiles as horizontal lines instead of a mini-box without split :
sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',inner = 'quartile')

# Draw the quartiles as horizontal lines instead of a mini-box with split:
sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',               split = True,inner = 'quartile')

# Show each observation with a stick inside the violin:

sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',               split = True,inner = 'stick')

# Scale the density relative to the counts across all bins:

sns.violinplot(x = 'day',y ='total_bill',data = tips_data,hue = 'sex',palette = 'muted',               split = True,inner = 'stick',scale_hue = False)

# Use a narrow bandwidth to reduce the amount of smoothing:


sns.violinplot(x="day", y="total_bill", hue="sex",                    data=tips_data, palette="Set2", split=True,                   scale="count", inner="stick",                   scale_hue=False, bw=.2)
plt.legend()

# Draw horizontal violins:
planets_data = sns.load_dataset("planets")

sns.violinplot(x="orbital_period", y="method",                   data=planets_data[planets_data.orbital_period < 1000],                  scale="width", palette="Set3")

# Don’t let density extend past extreme values in the data:

sns.violinplot(x="orbital_period", y="method",                   data=planets_data[planets_data.orbital_period < 1000],                  scale="width", palette="Set3",cut =0)



