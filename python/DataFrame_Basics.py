import pandas as pd
from pandas import DataFrame

df = DataFrame([[1,2,3,4]], columns=["A", "B", "C", "D"])
df

rows = []
for e in range(0,5):
    rows.append([e+1,e+2,e+3,e+4])
    
df = DataFrame(rows, columns=["A", "B", "C", "D"])
df

df.index

for e in df.index:
    print(e)

df.index.tolist()

df.columns

df.columns.tolist()

df.head(2)

df.tail(2)

df.sum()

df.A.sum()

df.A.mean()

df.median()

df.describe()

df

df.index.tolist()

df.drop(2)

df.drop([0,3])

ser = df.ix[0]
ser

type(ser)

ser["A"]

ser.keys()

df.ix[0]["A"].dtype

df

df.loc[1, "B"]

df.loc[1, "B"] = 100
df

for index, row in df.iterrows():
    print("ROW AT INDEX %d:\n%s\n\n" % (index, row))

df2 = df[df.A == 4]
df2  # copy with the selected data

df  # original DataFrame

df.loc[3, "C"] = 50
df

df2

df = DataFrame([
        ["Test", "Another Test"],
        ["String", "Another String"],
    ], columns=["A", "B"])
df

df[df.A.str.contains("Test")]

# convert the rows to dictionaries that include the index of the row
df2.to_dict()

# convert the rows to flat lists
df2.to_dict(orient="list")

# copy the last table output within this workbook
clipboard_df = pd.read_clipboard()
clipboard_df

html_dfs = pd.read_html("http://www.cisco.com/c/en/us/products/collateral/switches/catalyst-2960-series-switches/eos-eol-notice-c51-730121.html")
type(html_dfs)

html_dfs[0]

html_dfs[1]



