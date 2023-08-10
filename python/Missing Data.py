import pandas as pd
get_ipython().magic('pylab inline')

df = pd.read_csv("../data/train.csv")

df[df.Age.isnull()]

#replace with mean age...
avgAge = df.Age.mean()

df.Age = df.Age.fillna(value=avgAge)

df[df.Age.isnull()]

df





