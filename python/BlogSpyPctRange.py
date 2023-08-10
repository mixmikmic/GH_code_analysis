import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# https://www.quandl.com/data/YAHOO/INDEX_GSPC-S-P-500-Index
spy = get_pricing(symbols("SPY"), fields=["close_price", "high", "low"],
                  start_date=pd.Timestamp("2002-01-01"), end_date=dt.date.today())

spy = pd.DataFrame(spy, index=spy.index)
spy.rename(columns={"close_price" : "close"}, inplace=True)
spy.drop(["high", "low"], inplace=True, axis=1)
spy.dropna(inplace=True)

def rets(values, shift):
    ret = (values.shift(-shift) / values) - 1
    return ret

spy["pct"] = spy["close"].pct_change()

spy.dropna(inplace=True)

spy[abs(spy["pct"]) >= 0.01].iloc[-1]

spy_reset = spy.copy().reset_index()
instances = []

start_index = [0]
end_index = [0]
for index, row in spy_reset.iterrows():
    abs_ret = abs(row["pct"])
    
    if abs_ret > 0.01:
        start_index.append(0)
    
    if abs_ret <= 0.01:
        end_index.append(start_index[-1])

    if abs_ret <= 0.01 and abs(spy_reset["pct"]).iloc[index-1] > 0.01:
        start_index.append(index)
    
    instances.append([start_index[-1], row["pct"]])
    
instances_df = pd.DataFrame(instances)
instances_df.columns = ["inst", "pct"]
instances_df = instances_df[instances_df["inst"] != 0]

instances_df

current_run = instances_df.groupby(["inst"]).get_group(3760)
current_run.loc[-1] = [0, 0]
current_run.index = current_run.index + 1
current_run = current_run.sort()
current_run.reset_index(inplace=True)


for index, group in instances_df.groupby(["inst"]):
    if len(group) > 10:
        group.loc[-1] = [0, 0]
        group.index = group.index +1
        group = group.sort()
        group.reset_index(inplace=True)
        plt.plot(group["pct"].cumsum(), color="#555555", alpha=0.42, linewidth=1, label="_nolegend_")
        
plt.plot(current_run["pct"].cumsum(), color="crimson", label="Current streak (Start, Dec 12 2016)")
plt.title("Spy low volatility streaks (+/-1%) of longer than 10 days", fontsize=11)
plt.xlabel("# Of trading days in low vol streak")
plt.ylabel("Streak return")
plt.legend(loc="center right")
plt.grid(alpha=0.21)

returns_df = spy.copy().reset_index()
returns_df["streak"] = np.array(instances)[:, 0]

returns_n = pd.DataFrame(index=np.arange(0, 100))

for index_g, group in returns_df.groupby(["streak"]):
    if len(group) >= 21:
        index_g = int(group.index[-1])
        out = returns_df.iloc[index_g:index_g+64]
        out = out.reset_index(drop=True)
        #plt.plot(out["pct"].cumsum())
        returns_n[index_g] = out["pct"].cumsum()
        

returns_n.loc[-1, returns_n.columns.values] = 0
returns_n.sort_index(inplace=True)
returns_n.reset_index(drop=True, inplace=True)
returns_n.dropna(how="all", inplace=True)

plt.plot(returns_n, color="#555555", alpha=0.42, linewidth=1, label="_nolegend_")
plt.plot(returns_n.mean(axis=1), color="crimson", label="Mean rets")
plt.title("Spy returns after a break of low volatility streak (+/-1%) of longer than 21 days", fontsize=11)
plt.yticks(np.arange(-0.16, 0.11, 0.02))
plt.xlim(0, 63)
plt.xlabel("# Of trading days after low vol streak ends")
plt.ylabel("Post streak return")
plt.legend(loc="upper left")
plt.grid(alpha=0.21)

































































