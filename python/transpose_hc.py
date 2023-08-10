import numpy as np
import pandas as pd

source_file = 'fadl.csv'

df = pd.read_csv(source_file, index_col=0)
df.head()

expected_rows = len(df.index)*len(df.columns)
print("Expected number of rows in the output: ", str(expected_rows))

source_sum = df.sum().sum()
print("Sum total of source data: ", str(source_sum))

stacked = df.stack()

stacked_rows = len(stacked)
if stacked_rows == expected_rows:
    print("So far so good. Expecting", str(expected_rows), "rows and transposed data frame has", str(stacked_rows), "rows.")
else:
    print("Error. Expecting", str(expected_rows), "rows and transposed data frame has", str(stacked_rows), "rows. Please check.")

cols = ['date_time', 'count']
output = pd.DataFrame(columns=cols)

for i in range(len(stacked)):
    tocat = stacked.index[i]
    dt = tocat[0]+":"+tocat[1]
    v = stacked.iloc[i]
    tmpdf = pd.DataFrame([[dt, v]], columns=cols)
    output = output.append(tmpdf)
    
output.head()

output_rows = len(output)

if output_rows == expected_rows:
    print("First check successful. Expecting", str(expected_rows), "rows and final data frame has", str(output_rows), "rows.")
else:
    print("Error. Expecting", str(expected_rows), " rows and final data frame has", str(output_rows), "rows. Please check.")

output_sum = output.sum()[1]

if output_sum == source_sum:
    print("Second check successful. Expecting sum total of", str(source_sum), "and final total is", str(output_sum), ".")
else:
    print("Error. Expecting sum total of", str(source_sum), "and final total is", str(output_sum), ".")

output_file = 'fadl_data.csv'
output.to_csv(output_file, index=False)

