import pandas as pd

df = pd.read_excel("/users/AkiraKaneshiro/Downloads/sales-funnel.xlsx")
df

df["Status"] = df["Status"].astype("category")
df["Status"].cat.set_categories(["won","pending","presented","declined"], inplace=True)

pd.pivot_table(df, index=["Name"])

pd.pivot_table(df, index=["Name","Rep","Manager"])

pd.pivot_table(df, index=["Manager","Rep"])

pd.pivot_table(df, index=["Manager","Rep"], values=["Price"]) ## Default by mean

import numpy as np
pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], aggfunc=np.sum)

pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], aggfunc=[np.sum, np.mean, len])

pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], aggfunc=[np.sum], columns=["Product"])

pd.pivot_table(df, index=["Manager","Rep"], values=["Price"], 
               aggfunc=[np.sum], columns=["Product"], fill_value=0)

pd.pivot_table(df, index=["Manager","Rep"], values=["Price", "Quantity"], 
               aggfunc=[np.sum], columns=["Product"], fill_value=0)

pd.pivot_table(df, index=["Manager","Rep","Product"], values=["Price", "Quantity"], aggfunc=[np.sum], fill_value=0)

pd.pivot_table(df, index=["Manager","Rep","Product"], values=["Price", "Quantity"], 
               aggfunc=[np.sum, np.mean], fill_value=0, margins=True)

pd.pivot_table(df, index=["Manager","Status"], values=["Price"], aggfunc=[np.sum], fill_value=0, margins=True)

pd.pivot_table(df, index=["Manager","Status"], values=["Price","Quantity"], columns=["Product"],
              aggfunc={"Quantity":len, "Price":np.sum}, fill_value=0)

table = pd.pivot_table(df, index=["Manager","Status"], values=["Price","Quantity"], columns=["Product"],
              aggfunc={"Quantity":len, "Price":[np.sum, np.mean]}, fill_value=0)
table

table.query("Manager == 'Debra Henley'")

table.query("Status == ['pending','won']")



