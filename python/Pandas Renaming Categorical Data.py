import pandas as pd

df = pd.DataFrame({"X1":["a","b","c","d","e","f","g","h", "i"], "X2":['Volvo', 'Skoda', 'Mazda', 'Ford', 'Mazda', 'Ford', 'Volvo', 'Skoda', 'Volvo' ], "X3":[110,120,130,140,150,160,170,180,190]})
df

df.dtypes

df['X2'].unique()

df["X2"] = pd.Categorical(df['X2'], df['X2'].unique())

df

df.dtypes

df["X2"]  = df["X2"].cat.rename_categories([1,2,3,4])
df

df["X2"] = df["X2"].cat.rename_categories(['Volvo', 'Skoda', 'Mazda', 'Ford'])
df

