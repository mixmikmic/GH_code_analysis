import pandas as pd

sales1 = {'account' : ['Jones LLC' , 'Alpha Co' , 'Blue Inc'],
         'Jan' : [150, 200, 50],
         'Feb' : [200, 210, 90],
         'Mar' : [140, 215, 95]}

df1 = pd.DataFrame(sales1)[['account', 'Jan', 'Feb', 'Mar']]
df1

df2 = pd.DataFrame.from_dict(sales1)[['account', 'Jan', 'Feb', 'Mar']]
df2

sales2 = {0 : ['Jones LLC', 150, 200, 140],
          1 : ['Alpha Co', 200, 210, 215],
          2 : ['Blue Inc', 50, 90, 95]}

df3 = pd.DataFrame.from_dict(sales2, orient='index')
df3.columns = ['acccount', 'Jan', 'Feb', 'Mar']
df3

