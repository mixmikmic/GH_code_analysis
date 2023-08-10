import pandas as pd
# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)

df

df.groupby('Company')

by_comp = df.groupby("Company")

by_comp.mean()

df.groupby('Company').mean()

by_comp.std()

by_comp.min()

by_comp.max()

by_comp.count()

by_comp.describe()

by_comp.describe().transpose()

by_comp.describe().transpose()['GOOG']

