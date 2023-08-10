import pandas as pd
df = pd.read_csv('data.csv')                       # pd.read_table('data.txt', sep=',')
df                                                 # pd.read_table('data.txt', sep='\s+')

# pd.read_csv('ch06/ex2.csv', header=None)         if the file contains no headers.

names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('data.csv', names=names, index_col='message')

parsed = pd.read_csv('data_multi.csv', index_col=['key1', 'key2'])
parsed

pd.read_csv('data1.csv', skiprows=[0, 2, 3])

result = pd.read_csv('data_null.csv')                # result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])  
result                                  # sentinels = {'message': ['foo', 'NA'], 'something': ['two']}

pd.read_csv('data.csv', nrows=5)         # to read just 5 rows.

df.to_csv('output.csv')                   # to output as CSV
df.to_csv(sys.stdout, sep='|')            # Print on console instead of file
df.to_csv(sys.stdout, na_rep='NULL')      # Replace NA with NULL
df.to_csv(sys.stdout, index=False, header=False)
df.to_csv(sys.stdout, index=False, cols=['a', 'b', 'c'])     # to write particular columns and in particular order

# pip install xlrd
xls_file = pd.ExcelFile('data.xls')
table = xls_file.parse('Sheet1')
table



