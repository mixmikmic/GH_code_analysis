import pandas as pd

df = pd.read_csv("http://www.electoralcommission.org.uk/__data/assets/file/0014/212135/EU-referendum-result-data.csv")

leave = df["Leave"].sum() 
remain = df["Remain"].sum()
print("{} - {} = {}".format(leave, remain, (leave - remain)))

leave / (remain + leave) *100

electorate = df['Electorate'].sum(); electorate

turnout = (leave + remain) / electorate * 100; turnout

rejected = df.Rejected_Ballots.sum(); rejected

get_ipython().magic('matplotlib inline')

dfa = df.groupby("Area").sum()
dfa.head()
dfa["Perc_leave"] = dfa["Leave"] / (dfa["Remain"] + dfa["Leave"]) * 100
dfa["Perc_remain"] = dfa["Remain"] / (dfa["Remain"] + dfa["Leave"]) * 100

dfa.head(3)

top5_leave = dfa[["Perc_leave", "Perc_remain"]].sort_values(by="Perc_leave", ascending=False)[0:5]
top5_leave.head()

plt1 = top5_leave.plot(kind="bar")
plt1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

top5_remain = dfa[["Perc_leave", "Perc_remain"]].sort_values(by="Perc_leave", ascending=False)[-5:]
top5_remain.head()

plt2 = top5_remain.plot(kind="bar")
plt2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

dfr = df.groupby("Region").sum()

# ok next time I should probably wrap this in a method
dfr["Perc_leave"] = dfr["Leave"] / (dfr["Remain"] + dfr["Leave"]) * 100
dfr["Perc_remain"] = dfr["Remain"] / (dfr["Remain"] + dfr["Leave"]) * 100

dfr[["Perc_leave", "Perc_remain"]].sort_values(by="Perc_leave", ascending=False).plot(kind="bar")

