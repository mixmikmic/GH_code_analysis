# This drill assumes you've loaded and cleaned the data from 6.7.4

# Creating a binary variable where 1 = never married  and 0 =  married at some point
data2["nevermarried"] = (data2["MARSTAT"] == 50).astype(int)



groups = data2.groupby("nevermarried")
ax = plt.axes()
married = ["never married","married"]
# Fitting a survival function for each group
for group in groups:
    sf = sm.SurvfuncRight(group[1]["Longevity"], group[1]["dead"])
    sf.plot(ax)
li = ax.get_lines()
plt.figlegend((li[0], li[2]), married, "center right")
ax.set_ylabel("Proportion alive")
ax.set_xlabel("Age")
ax.set_autoscaley_on(False)
ax.set_ylim([.85,1])

yearmod = smf.phreg("Longevity ~  female + nevermarried", # The model
                data2, # The data
                status=data2['dead'].values # Whether values are right-censored
                ) 
yrrslt = yearmod.fit()
print(yrrslt.summary())



