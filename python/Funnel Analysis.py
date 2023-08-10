import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use("ggplot")
from sklearn.feature_selection import chi2

# Load the data
allusers = pd.read_csv("user_table.csv", index_col="user_id")
allusers.head()

user_to_search = pd.read_csv("search_page_table.csv", index_col="user_id")
user_to_search.head()

user_to_pay = pd.read_csv("payment_page_table.csv", index_col="user_id")
user_to_pay.head()

user_to_confirm = pd.read_csv("payment_confirmation_table.csv", index_col="user_id")
user_to_confirm.head()

allusers.loc[user_to_search.index, "page"] = user_to_search["page"]
allusers.head()

allusers.loc[user_to_pay.index, "page"] = user_to_pay["page"]
allusers.tail(10)

allusers.loc[user_to_confirm.index, "page"] = user_to_confirm["page"]

# Let's change the name of page in all users to a better name
allusers.rename(columns={"page":"final_page"}, inplace=True)
allusers["final_page"].fillna("home_page", inplace=True)

# change date object to pandas datetime object
allusers["date"] = pd.to_datetime(allusers["date"])
allusers.head()

allusers.info()

allusers.to_csv("all_users.csv",index_label="user_id")

allusers.groupby("device")["final_page"].apply(lambda s:s.value_counts()).unstack()

allusers.groupby('device')["final_page"].apply(lambda s: s.value_counts(normalize=True)).unstack()

allusers.head()

allusers.groupby("sex")["final_page"].value_counts()

X = allusers.copy()

X["device"].value_counts()

X["from_mobile"] = (X["device"] == "Mobile").astype(int)
del X["device"]
# for simplicity let's drop date
del X["date"]

X["is_male"] = (X["sex"] == "Male").astype(int)
del X["sex"]

X["converted"] = (X["final_page"] == "payment_confirmation_page").astype(int)
del X["final_page"]

X["converted"].mean() * 100 # highly imbalanced classification problem

X.head()

X.describe()

# impact of sex
X.groupby("is_male")["converted"].agg(["count", "mean"]).sort_values(by="mean", ascending=False)

# Statistical Significance
X = X
y = X.pop("converted")

scores, pvalues = chi2(X,y)

pd.DataFrame({"chi2_score":scores, "p-value":pvalues}, index=X.columns).sort_values(by="chi2_score", ascending=False)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
dt = DecisionTreeClassifier()
dt.fit(X,y)
dot_data = export_graphviz(dt, out_file=None, 
                           feature_names=X.columns, 
                           class_names=["NotConverted", "Converted"],
                           proportion=True, 
                           leaves_parallel=True, filled=True)

