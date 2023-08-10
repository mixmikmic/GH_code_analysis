import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

df = pd.read_csv("indicators_info.csv")
df.head()

len(df)

sns.set(rc={"figure.figsize": (10, 6)})
ax = sns.distplot(df["FirstYear"], kde=False, rug=True)
ax.set_title('First Years')

ax = sns.distplot(df["LastYear"], kde=False, rug=True)
ax.set_title('Last Years')

ax = sns.distplot(df["NumYears"], kde=False, rug=True, bins=20)
ax.set_title('Number of Years')

ax = sns.distplot(df["NumCountries"], kde=False, rug=True, bins=20)
ax.set_title('Number of Countries')

print("Indicators used by 200+ countries: {}".format(sum(df["NumCountries"] >= 200)))
print("Indicators around for 35+ years: {}".format(sum(df["NumYears"] >= 35)))
print("Indicators used by 200+ countries AND 35+ years: {}".format(
        len(df[(df["NumCountries"] >= 200) & (df["NumYears"] >= 35)])))

print("Indicators used by 200+ countries for < 35 years: {}".format(
        len(df[(df["NumCountries"] >= 200) & (df["NumYears"] < 35)])))
print()
print(df[(df["NumCountries"] >= 200) & (df["NumYears"] < 35)][["IndicatorName", "NumYears"]].to_string())

print(df[(df["NumCountries"] >= 200) & (df["NumYears"] >= 35)]["IndicatorName"].to_string())

inds = list(df[(df["NumCountries"] >= 200) & (df["NumYears"] >= 35)]["IndicatorCode"])
inds

# Save indicators
df[(df["NumCountries"] >= 200) & (df["NumYears"] >= 35)].to_csv("data/top_inds_list.csv")

f_inds_df = pd.read_csv("data/female_inds.csv")
f_inds_df.head()

f_inds = list(f_inds_df["IndicatorCode"])
f_inds.append('NY.GDP.PCAP.CD') # Add GDP per capita: to predict

full_inds_df = pd.read_csv("data/Indicators.csv")
full_inds_df.head()

len(full_inds_df)

# grab all data about the specified female edu/employment indicators
inds_df = full_inds_df[full_inds_df["IndicatorCode"].isin(f_inds)]
inds_df.to_csv("data/indicators_female.csv")
len(inds_df)

# create reference map: {Indicator code: indicator name}
ind_name = {}
for i, row in f_inds_df.iterrows():
    ind_name[row["IndicatorCode"]] = row["IndicatorName"]

def get_value(df, code, year):
    if not any((df["IndicatorCode"] == code) & (df["Year"] == year)):
        return float("nan")
    return df.loc[(df["IndicatorCode"]==code) & (df["Year"]==year), "Value"].item()

# get df for country where rows=years and columns=[edu inds] 

def get_country_inds(country):
    country_df = inds_df[inds_df["CountryCode"]==country]
    data = {}
    for ind in f_inds:
        data[ind] = [get_value(country_df, ind, year) for year in range(1980, 2014)]
        data["year"] = [year for year in range(1980, 2014)]
    return data

def get_country_df(country):
    return pd.DataFrame(get_country_inds(country))

us_df = get_country_df("USA")
us_df.head()

country_list = set(inds_df["CountryCode"])
len(country_list)

# get indicator data (in map form) for each country
f_inds_map = []
for country in country_list:
    f_inds_map.append(get_country_inds(country))

# put together above data into one dataframe, dropping all rows without a GDP per capita value
full_df = pd.concat([pd.DataFrame(c) for c in f_inds_map], ignore_index=True)
full_df = full_df[pd.notnull(full_df['NY.GDP.PCAP.CD'])]
len(full_df)

full_df.to_csv("data/female_indicators_missing.csv")

# impute missing values from df
from fancyimpute import BiScaler, KNN, SoftImpute

df_imputed = KNN(k=3).complete(full_df)
df_imputed

df_imputed = pd.DataFrame(df_imputed, columns=full_df.columns)

df_imputed.to_csv("data/knn_imputed_indicators.csv", index=False)

