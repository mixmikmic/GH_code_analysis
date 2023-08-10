import pandas as pd
import numpy as np

df = pd.read_csv("dataset/titanic.csv")
df

df.head()

df.columns.values

# To apply some function on index 
df.index.map(lambda x: x+1)

# To sum all coulmns
df.sum(skipna=True,axis=1)

# To get a data summary
df.describe()

# Get detailed info
df.info()

df.info(memory_usage='deep')

df.memory_usage()

# To delete a column
df.drop(["PassengerId"],axis=1,inplace=False)
#To delete a row
df.drop(0,axis=0)

df["Pclass"].unique()

df.rename(columns={"PassengerId":"Id","Fare":"Amount"},index={})

# To get columns 
df[["Fare","Survived"]]

#Get a cell
df["Fare"][3]

# Get rows
df[2:6]

df.loc[0:4,:]

df.loc[0:5,"Name":"Ticket"]

df_new = pd.DataFrame(np.random.randn(4,4), columns=list('ABCD'), index=list('abcd'))
df_new
df_new.loc["c":,"A":"C"]

#Get Cell
df_new.loc["c","A"]

df.iloc[2,:]

df.iloc[2:5,0:3]

df_new = pd.DataFrame(np.random.randn(4,4), columns=list('ABCD'), index=list('abcd'))
df_new
df_new.ix[1:4,"A":"C"]

df_new.ix["c":"d",0:]

df[(df["Age"]>50) & (df["Sex"]=="male")]

df[(df["Fare"]>50 )| (df["Embarked"]=="S")]

df.groupby(["Pclass"]).groups.keys()

df.groupby(["Pclass","Sex"]).groups.keys()

df.groupby("Sex").get_group("male")

df.groupby("Pclass").count()

df.groupby("Pclass").mean()

df.groupby("Pclass").agg(np.median)

df.groupby("Pclass").agg([np.sum,np.mean])

df.groupby("Pclass").agg({"Fare":lambda x : np.median(x),
                           "Age":np.median})



