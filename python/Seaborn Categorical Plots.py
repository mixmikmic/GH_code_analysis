import seaborn as sns
get_ipython().magic('matplotlib inline')

tips = sns.load_dataset('tips')
tips.head()

sns.barplot(x='sex',y='total_bill',data=tips)

import numpy as np

sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)

sns.countplot(x='sex',data=tips)

sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')

# Can do entire dataframe with orient='h'
sns.boxplot(data=tips,palette='rainbow',orient='h')

sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")

sns.violinplot(x="day", y="total_bill", data=tips,palette='rainbow')

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',palette='Set1')

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')

sns.stripplot(x="day", y="total_bill", data=tips)

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True)

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1')

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True)

sns.swarmplot(x="day", y="total_bill", data=tips)

sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True)

sns.violinplot(x="tip", y="day", data=tips,palette='rainbow')
sns.swarmplot(x="tip", y="day", data=tips,color='black',size=3)

sns.factorplot(x='sex',y='total_bill',data=tips,kind='bar')

