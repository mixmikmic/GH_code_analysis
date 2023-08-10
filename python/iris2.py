import pandas as pd

iris=pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv",header=None)

iris.head()

iris2=iris.iloc[1:,1:]

iris2.head()

import os as os

os.getcwd()

iris2.to_csv("iris2.csv")

iris.info()

os.getcwd()

os.chdir('C:\\Users\\Dell\\Desktop')

os.listdir()



