import pandas as pd

asd = pd.read_csv("data/input.csv")
print type(asd) 
asd.head()
# This is a Dataframe because we have multiple columns!

data = pd.read_csv("data/input.csv", usecols=["name"], squeeze=True)
print type(data)
data.head()
data.index

data = pd.read_csv("data/input_with_one_column.csv", squeeze=True)
print type(data)

# HEAD
print data.head(2), "\n"
# TAIL
print data.tail()

list(data)

dict(data)

max(data)

min(data)

dir(data)

type(data)

sorted(data)

data = pd.read_csv("data/input_with_two_column.csv", index_col="name", squeeze=True)
data.head()

data[["Alex", "asd"]]

data["Alex":"Vale"]



