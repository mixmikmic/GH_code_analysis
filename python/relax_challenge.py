import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

user_engagement = pd.read_csv("takehome_user_engagement.csv")
user_engagement.head()

users = pd.read_csv("takehome_users.csv", encoding = "latin1")
users.head()

user_engagement["datetime"] = pd.to_datetime(user_engagement.time_stamp)
user_engagement_counts = user_engagement["user_id"].value_counts()
user_engagement_3_or_more = user_engagement[user_engagement["user_id"].isin(user_engagement_counts[user_engagement_counts > 2].index)]
user_engagement_3_or_more.head()

adopted_users = []
for this_user in user_engagement_3_or_more["user_id"].unique():
    this_users_engagement = user_engagement_3_or_more[user_engagement_3_or_more["user_id"] == this_user]
    this_users_datetime = this_users_engagement["datetime"].reset_index()["datetime"]
    for i in range(len(this_users_datetime) - 2):
        time_interval = this_users_datetime[i + 2] - this_users_datetime[i]
        if time_interval < pd.Timedelta("7 days"):
            adopted_users.append(this_user)
            break

len(adopted_users)

user_adoption_ind = users["object_id"].isin(adopted_users)
pd.crosstab(user_adoption_ind, users["creation_source"], normalize = "columns")

pd.crosstab(user_adoption_ind, users["opted_in_to_mailing_list"], normalize = "columns")

pd.crosstab(user_adoption_ind, users["enabled_for_marketing_drip"], normalize = "columns")

pd.crosstab(user_adoption_ind, users["org_id"], normalize = "columns")

users["org_id"].value_counts().head(10)

fisher_exact(pd.crosstab(user_adoption_ind, users["org_id"]).iloc[:, [0, 9]].values)

adopted_years = pd.to_datetime(users["creation_time"][user_adoption_ind]).apply(lambda x: x.year).value_counts()
nonadopted_years = pd.to_datetime(users["creation_time"][~user_adoption_ind]).apply(lambda x: x.year).value_counts()

adopted_years/(adopted_years + nonadopted_years)

pd.to_datetime(users["creation_time"]).apply(lambda x: x.year).value_counts()

adopted_months = pd.to_datetime(users["creation_time"][user_adoption_ind]).apply(lambda x: x.month).value_counts()
nonadopted_months = pd.to_datetime(users["creation_time"][~user_adoption_ind]).apply(lambda x: x.month).value_counts()

adopted_months/(adopted_months + nonadopted_months)

