import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import quandl
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set(style="whitegrid")

sdate = pd.Timestamp("2017-01-01")

dj = pd.read_csv("../Data/DjiaHist.csv", index_col = "Date")
dj.index = pd.to_datetime(dj.index)
dj.sort_index(ascending=True, inplace=True)
dj.rename(columns={"Value" : "value"}, inplace=True)
dia = quandl.get("YAHOO/INDEX_DJI", authtoken="ZW3mrFD6ft-vHhH_Fs1y")
dia.rename(columns={"Close":"price"}, inplace=True)
dia.drop(["Open", "High", "Low", "Volume", "Adjusted Close"], axis=1, inplace=True)
dia = dia.loc[sdate:]

dj["pct"] = np.log(dj["value"]).diff()
dj["year"] = dj.index.year
dj["day"] = dj.index.dayofyear

dia["day"] = dia.index.dayofyear
dia["pct"] = np.log(dia["price"]).diff()
dia.set_index(dia["day"], inplace = True)
dia.fillna(0, inplace=True)

daily_rets = pd.pivot_table(dj, index=["day"], columns=["year"], values=["pct"])
#daily_rets.convert_objects(convert_numeric = True)
#pd.to_numeric(daily_rets, inplace=True)
daily_rets.fillna(0, inplace = True)
daily_rets.columns = daily_rets.columns.droplevel()
daily_rets.drop(2016, axis =1, inplace = True)
daily_rets.rename(columns = lambda x: str(x), inplace=True)

daily_rets.head(8)

f, ax = plt.subplots(figsize=(18, 12))
ax.plot(daily_rets.cumsum(), color="#333333", linewidth=1, alpha=0.1, label=None)
ax.plot(dia["pct"].cumsum(), linewidth=2, color="crimson", label="2017 returns")
plt.grid(False)
plt.ylabel("Annual return")
plt.xlabel("Day of the year")
plt.ylim(-0.7, 0.7)
plt.xlim(0, 365)
plt.axhline(0, linewidth= 1, color="#333333", linestyle="--")
plt.legend(loc="upper left")





#ax.plot(daily_rets.index, daily_rets.cumsum(), color="#333333", linewidth=1, alpha=0.06, label=None)
#ax.plot(daily_rets["mean"].cumsum(), color="#333333", linewidth=2, alpha=0.8, label="Mean returns since 1896")
#ax.plot(daily_rets["el_year"].cumsum(), color="g", linewidth=2, alpha=0.8, label="Election year mean returns since 1936")


daily_rets["mean"] = daily_rets.mean(axis=1)
daily_rets["2016"] = dia["pct"]
daily_rets["el_year"] = daily_rets.loc[:, "1936"::4].mean(axis=1)
daily_rets["post_el"] = daily_rets.loc[:, "1937"::4].mean(axis=1)


plt.figure(figsize=(18, 12))

ax1 = plt.subplot2grid((4,1), (0,0), rowspan=3)
ax1.plot(daily_rets.index, daily_rets.cumsum(), color="#333333", linewidth=1, alpha=0.06, label=None)
ax1.plot(daily_rets["mean"].cumsum(), color="#333333", linewidth=2, alpha=0.8, label="Mean returns since 1896")
ax1.plot(daily_rets["post_el"].cumsum(), color="g", linewidth=2, alpha=0.8, label="Post election year mean returns since 1936")
ax1.plot(daily_rets["2016"].dropna().cumsum(), linewidth=2, color="crimson", label ="2017 returns")

plt.title("Cumulative Dow 2017 Returns Vs Mean Historical Returns Since 1896")
plt.axhline(0, linewidth= 1, color="#333333", linestyle="--")
plt.axvline(dia.index[-1], linestyle="--", color="#555555")
plt.ylim(-0.15, 0.15)
plt.grid(False)
plt.legend(loc="upper left")

ax2 = plt.subplot2grid((4,1), (3,0), rowspan=3, sharex=ax1)
ax2.fill_between(daily_rets.index, 0, daily_rets["mean"], where= daily_rets["mean"]<0, color="crimson")
ax2.fill_between(daily_rets.index, daily_rets["mean"], 0, where= daily_rets["mean"]>0, color="forestgreen")

plt.axvline(dia.index[-1], linestyle="--", color="#555555")
plt.title("Mean Daily Returns")
ax2.grid(False)
plt.xlim(1, 365)

def decadeMean(start, end):
    return daily_rets.loc[:, start : end].cumsum().mean(axis=1)

decade_rets = pd.DataFrame({#"1900’s" : decadeMean("1900", "1909"),
                            #"1910’s" : decadeMean("1910", "1919"),
                            #"1920’s" : decadeMean("1920", "1929"),
                            #"1930’s" : decadeMean("1930", "1939"),
                            #"1940’s" : decadeMean("1940", "1949"),
                            #"1950’s" : decadeMean("1950", "1959"),
                            #"1960’s" : decadeMean("1960", "1969"),
                            "1970’s" : decadeMean("1970", "1979"),
                            "1980’s" : decadeMean("1980", "1989"),
                            "1990’s" : decadeMean("1990", "1999"),
                            "2000’s" : decadeMean("2000", "2009"),
                            "2010’s" : decadeMean("2010", "2015")
                           }, index= daily_rets.index)

mean_rets = decade_rets.rolling(21).mean()


plt.figure(figsize=(18, 12))
mean_rets.plot(linewidth=1)
rets_df["pct"].dropna().cumsum().rolling(21).mean().plot(color="crimson", linewidth=2, label="2016")
plt.legend(loc="upper left")
plt.ylabel("Annual return")
plt.xlabel("Day of the year")
plt.grid(False)

old = daily_rets["1910"]
new = daily_rets["2015"]

plt.figure(figsize=(18, 8))
plt.plot(np.random.permutation(old).cumsum(), label="Year X (1910)")
plt.plot(np.random.permutation(new).cumsum(), label="Year Y (2015)")
plt.legend(loc="upper left")
plt.xlim(0, 365)
plt.ylabel("Annual return")
plt.xlabel("Day of the year")
plt.grid(False)











