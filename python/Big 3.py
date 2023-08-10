#Install libraries needed
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')

#Import libraries needed
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv("uslisted.txt",encoding="utf-16",sep="\t",na_values = ["-","n.a."],thousands=",")
df = df.loc[df["Stock exchange(s) listed"].
            isin([ 'NYSE MKT','NYSE ARCA','NASDAQ/NMS (Global Market)','NASDAQ National Market',
                  'New York Stock Exchange (NYSE)'])]
df = df.drop_duplicates(subset="BvD ID number")
c= Counter(df["Type of entity"])
c

df = pd.read_csv("uslisted.txt",encoding="utf-16",sep="\t",na_values = ["-","n.a."],thousands=",")
df = df.loc[df["Stock exchange(s) listed"].isin([ 'NYSE MKT','NYSE ARCA','NASDAQ/NMS (Global Market)','NASDAQ National Market','New York Stock Exchange (NYSE)'])]
df = df.loc[df["Type of entity"].isin(['Foundation/Research institute', 'Bank', 'Venture capital', 'Financial company', 
                                'Industrial company', 'Insurance company']),:]
df = df.loc[df["Shareholder - BvD ID number"] != "US041867445",:]
df = df.loc[df["Shareholder - Name"] != "PUBLIC",:]


companies = df.loc[:,["Company name","BvD ID number","Operating revenue (Turnover) th USD Last avail. yr",
                "Total assets (last value) th USD",'Number of employees Last avail. yr',
                "Current market capitalisation th USD","Stock exchange(s) listed","Type of entity"]].drop_duplicates()
ownership = df.loc[:,["Company name","BvD ID number","Shareholder - Name","Shareholder - BvD ID number",
                "Shareholder - Direct %","Shareholder - Total %"]]

        

d = {np.NaN: np.NaN,"NG": 0.01,"MO": 50.01, "WO": 98.01, "GP": 50.01,">50.00":50.01}


#NG:0 MO: 50.01 WO: 98.1 GP"
for i in sorted([_ for _ in set(ownership["Shareholder - Direct %"]) | set(ownership["Shareholder - Total %"]) if isinstance(_,str)]):
    if not d.get(i): 
        try:
            d[i] = float(i)
        except:
            d[i] = float(i[1:])
ownership["Shareholder - Direct %"] = ownership["Shareholder - Direct %"].apply(lambda x: d[x])
ownership["Shareholder - Total %"] = ownership["Shareholder - Total %"].apply(lambda x: d[x])

ownership["max"] = ownership.apply(lambda x: np.nanmax([x["Shareholder - Direct %"],x["Shareholder - Total %"]]),axis=1)

#"US041867445"
with open("big3_position.csv","w+") as f:
    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".
            format("Company_name","Company_ID","Big3Share","Position","Revenue","Assets","Employees","MarketCap","Exchange","TypeEnt"))
    for id,g in ownership.groupby("BvD ID number"):
        sum_big3 = g.loc[g["Shareholder - BvD ID number"].isin(['US149144472L', 'US320174431', 'US042456637']),"max"].sum()
        t = g.loc[g["Shareholder - BvD ID number"] != "US041867445",:].sort_values(by="max",ascending=False,na_position="last")
        if sum_big3 == 0: position = 100
        else: position = 1
        for i,values in t.iterrows():
            if isinstance(values["Shareholder - BvD ID number"],float): continue
            if values.values[3][:2] != "ZZ":
                if values.values[-1] >=sum_big3: position+=1
                else: break
        r,a,e,m,exchange,typeent = companies.loc[companies["BvD ID number"] == values["BvD ID number"],:].values[0][-6:]
        #print(companies.loc[companies["BvD ID number"] == values["BvD ID number"],:].values[0])
        
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(values["Company name"],values["BvD ID number"],sum_big3,position,r,a,e,m,exchange,typeent))
        

df = pd.read_csv("big3_position.csv",sep="\t")
from collections import Counter
c = Counter(df["Position"])

sumc = np.sum([c[_] for _ in range(0,1007)])/100
print("Percentage of companies and percentage of market capitalization")
print(c[1]/sumc,100*np.sum(df.loc[df["Position"] == 1,"MarketCap"]/np.sum(df["MarketCap"])))
print(c[2]/sumc,100*np.sum(df.loc[df["Position"] == 2,"MarketCap"]/np.sum(df["MarketCap"])))
print(c[3]/sumc,100*np.sum(df.loc[df["Position"] == 3,"MarketCap"]/np.sum(df["MarketCap"])))
print((np.sum([c[_] for _ in range(4,1007)])/sumc),
      (100*np.sum(df.loc[~df["Position"].isin([1,2,3]),"MarketCap"]/np.sum(df["MarketCap"]))))
print()
print("Number of companies and capitalization (billions)")
print(c[1],np.sum(df.loc[df["Position"] == 1,"MarketCap"])*1000/1E9)
print(c[2],np.sum(df.loc[df["Position"] == 2,"MarketCap"])*1000/1E9)
print(c[3],np.sum(df.loc[df["Position"] == 3,"MarketCap"])*1000/1E9)
print(sumc*100-c[1]-c[2]-c[3], (np.sum(df["MarketCap"])
      -np.sum(df.loc[df["Position"] == 1,"MarketCap"])
      -np.sum(df.loc[df["Position"] == 2,"MarketCap"])
      -np.sum(df.loc[df["Position"] == 3,"MarketCap"]))*1000/1E9)
print()

c.most_common(10)

print("Ownership of each member of the big three and sum of means")
for i in [1,2,3]:
    df2 = df.loc[df["Position"] == i,:]
    o = pd.merge(ownership,companies,on="BvD ID number")
    o = pd.merge(o,df2,left_on="BvD ID number",right_on="Company_ID")
    o["x"] = o["max"]*o["Current market capitalisation th USD"]

    v = o.loc[o["Shareholder - BvD ID number"] == 'US149144472L',"max"].mean() #V
    b = o.loc[o["Shareholder - BvD ID number"] ==  'US320174431',"max"].mean() #BLK
    s = o.loc[o["Shareholder - BvD ID number"] == 'US042456637',"max"].mean() #SS

    print(1*v,1*b,1*s,1*(v+b+s))

df2 = df.loc[~df["Position"].isin([1,2,3]),:]
o = pd.merge(ownership,companies,on="BvD ID number")
o = pd.merge(o,df2,left_on="BvD ID number",right_on="Company_ID")
o["x"] = o["max"]*o["Current market capitalisation th USD"]

v = o.loc[o["Shareholder - BvD ID number"] == 'US149144472L',"max"].mean() #V
b = o.loc[o["Shareholder - BvD ID number"] ==  'US320174431',"max"].mean() #BLK
s = o.loc[o["Shareholder - BvD ID number"] == 'US042456637',"max"].mean() #SS

print(1*v,1*b,1*s,1*(v+b+s))
print()

print("Mean of sum of Ownership of each member of the big three")
for i in [1,2,3]:
    print(1*df.loc[df["Position"] == i,"Big3Share"].mean())
    
print(1*df.loc[~df["Position"].isin([1,2,3]),"Big3Share"].mean())

df = pd.read_csv("big3_position.csv",sep="\t")

endogenous_own = ownership.copy()
                                                                                  
endogenous_own = endogenous_own.loc[endogenous_own["max"]>=3.]
endogenous_own = endogenous_own.loc[endogenous_own["BvD ID number"] != endogenous_own["Shareholder - BvD ID number"],:]
edges = endogenous_own.loc[:,["BvD ID number","Shareholder - BvD ID number","max"]]

edges.columns = ["Source","Target","Weight"]
edges["Type"] = "Directed"


e1 = endogenous_own[["BvD ID number","Company name"]]
e1.columns = ["Id","Label"]
e2 = endogenous_own[["Shareholder - BvD ID number","Shareholder - Name"]]
e2.columns = ["Id","Label"]
nodes = pd.concat([e1,e2]).drop_duplicates()
nodes = pd.merge(nodes,df,left_on="Id",right_on="Company_ID")

nodes = nodes[["Id","Label","Position","Exchange","TypeEnt"]]

for i in range(4,1000):
    d[i] = 4
d[1] = 1
d[2] = 2
d[3] = 3
    

nodes["Position"] = nodes["Position"].apply(lambda x: d[x])

nodes.loc[nodes["Id"] == "US320174431","Position"] = 0
nodes.loc[nodes["Id"] == "US042456637","Position"] = 0
#VAnguard and FMR added by hand to have their names in the file
nodes.loc[122312] = ["US149144472L","VANGUARD INC via its funds",0,"None","Bank"]
nodes.loc[122313] = ["US126246544L","FMR LLC",0,"None","Bank"]

nodes.to_csv("nodes_allmarket.csv",sep="\t",index = None)
edges.to_csv("edges_allmarket.csv",sep="\t",index = None)





