import pandas as pd
get_ipython().magic('pylab inline')

df = pd.read_csv("../data/train.csv")

df.columns

df

df.shape

df.Survived.value_counts()

342/891.0

df.Sex.value_counts()

df.Sex.value_counts().plot(kind='bar')

df[df.Sex=='female']

df[df.Sex.isnull()]

df.Fare.value_counts()

df.describe()

df.Fare.hist(bins=5)

df[df.Fare.isnull()]

df[df.Fare==0]

df[df.Cabin.isnull()]

#women and children first?

fig, axs = plt.subplots(1,2)
df[df.Sex=='male'].Survived.value_counts().plot(kind='barh',ax=axs[0], title="Male Survivorship")
df[df.Sex=='female'].Survived.value_counts().plot(kind='barh',ax=axs[1], title="Female Survivorship")

df[df.Age <15].Survived.value_counts().plot(kind='barh')

df[(df.Age <15) &(df.Sex=='female')].Survived.value_counts().plot(kind='barh')

df[(df.Age <15) &(df.Sex=='male')].Survived.value_counts().plot(kind='barh')

for column in df.columns:
    if pd.isnull(df[column]).any():
        print(column)

df.head()

for column in df.columns:
    print(column)





