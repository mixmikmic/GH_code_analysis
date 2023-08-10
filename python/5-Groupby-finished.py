import pandas as pd

data = {'Company':['GOOG','MSFT','MSFT','GOOG','FB','FB'],
       'Person':['Sam','Kat','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,150,340,124,243,350]}

df = pd.DataFrame(data)

df

df.groupby('Company').groups

by_comp = df.groupby('Company')

by_comp.mean()

by_comp.std()

by_comp.min()

by_comp.max()

by_comp.count()

by_comp.describe()

by_comp.describe().transpose()

by_comp.describe().transpose()['GOOG']

by_comp.describe().T



