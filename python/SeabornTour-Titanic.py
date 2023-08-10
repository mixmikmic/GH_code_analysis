import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', 500)

df = sns.load_dataset('titanic')

# Write the code to look at the head of the dataframe

# Create a histogram to examine age distribution of the passengers.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['age'], bins = 10, range = (df['age'].min(),df['age'].max()))
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.show()

# What is a factorplot? Check the documentation! Which data are we using? What is the count a count of?

g = sns.factorplot("alive", col="deck", col_wrap=4, 
                   data=df[df.deck.notnull()], kind="count", size=4, aspect=.8)

# Try your own variation of the factorplot above.

# Draw a nested barplot to show survival for class and sex
g = sns.factorplot(x="CHANGE TO THE CORRECT FEATURE", 
                   y="CHANGE TO THE CORRECT FEATURE", 
                   hue="CHANGE TO THE CORRECT FEATURE", 
                   data=df,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")

g = sns.factorplot(x="CHANGE TO THE CORRECT FEATURE", 
                   y="CHANGE TO THE CORRECT FEATURE", 
                   col="CHANGE TO THE CORRECT FEATURE", 
                   data=df, 
                   saturation=.5, kind="bar", ci=None,aspect=.6)
(g.set_axis_labels("", "Survival Rate").set_xticklabels(["Men", "Women", "Children"]).set_titles
 ("{col_name} {col_var}").set(ylim=(0, 1)).despine(left=True)) 

# With factorplot, make a violin plot that shows the age of the passengers at each embarkation point 
# based on their class. Use the hue parameter to show the sex of the passengers

df.age = df.age.fillna(df.age.mean())

g = sns.pairplot(data=df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']], hue='survived', dropna=True)

# Pairplot of the crash data

g = sns.jointplot("fare", "age", df)

# Jointplot, titanic data

# Jointplot, crash data

#  boxplot of the age distribution on each deck by class

#  boxplot of the age distribution on each deck by class using FacetGrid

