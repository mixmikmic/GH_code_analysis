import pandas as pd
import os

cwd = os.getcwd()
fileName = 'example.csv'
filePath = os.path.join(cwd, fileName)

df = pd.read_csv(filePath,header=0)
print(df)

df['Time (s) - Dev1/ai0']

try:
    df[1]
except KeyError:
    print("KeyError: 1 - not a valid key")

cols = df.columns
for col in cols:
    print(df[col])



