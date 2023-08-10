import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

variables=["NCESSCH","LEAID","LEANM09","SCHNAM09","LSTATE09","LEVEL09","TYPE09",           "STATUS09","ULOCAL09","FTE09","TITLEI09","STITLI09","MAGNET09","CHARTR09",           "SHARED09","TOTFRL09","MEMBER09","WHITE09","TOTETH09"]

school=pd.read_table("data/ccd/2009-10/sc092a.txt")[variables]

#Remove "09" from variable name suffixes
vardict = {}
for variable in variables:
    if variable[-2:]=="09":
        vardict[variable]=variable[:-2]
school.rename(columns = vardict, inplace=True)
        
##Subset to only regular schools operational in 2009-10
school=school[(school.TYPE==1) & (school.STATUS.isin([1,3,4,5,8]))]
print school.shape

def pool(df, pooled, unpooled, values):
    df[pooled]=0
    df.ix[df[unpooled].isnull(), pooled]=np.nan
    df.ix[df[unpooled].isin(values), pooled]=1

pool(school, "NEW_ENGLAND","LSTATE",["CT","ME","MA","NH","RI","VT"])
pool(school, "MID_ATLANTIC","LSTATE",["NJ","NY","PA"])
pool(school, "EAST_NORTH_CENTL","LSTATE",["IL","IN","MI","OH","WI","AE"])
pool(school, "WEST_NORTH_CENTL","LSTATE",["IA","KS","MN","MO","NE","ND","SD"])
pool(school, "SOUTH_ATLANTIC", "LSTATE",["DE","DC","FL","GA","MD","NC","SC","VA","WV","VI","PR"])
pool(school, "EAST_SOUTH_CENTL", "LSTATE",["AL","KY","MS","TN"])
pool(school, "WEST_SOUTH_CENTL", "LSTATE",["AR","LA","OK","TX"])
pool(school, "MOUNTAIN", "LSTATE",["AZ","CA","CO","ID","MT","NV","NM","UT","WY"])    
pool(school, "PACIFIC", "LSTATE", ["AK","HI","OR","WA","AP","AS","MP","GU"])

print school[["NEW_ENGLAND","MID_ATLANTIC","EAST_NORTH_CENTL","WEST_NORTH_CENTL",                     "SOUTH_ATLANTIC","EAST_SOUTH_CENTL","WEST_SOUTH_CENTL",             "MOUNTAIN","PACIFIC"]].sum(axis=1).value_counts(dropna=False)

school["ULOCAL"]=pd.to_numeric(school.ULOCAL, errors="coerce")

pool(school, "CITY",  "ULOCAL",[11,12,13])
pool(school, "SUBURB","ULOCAL",[21,22,23])
pool(school, "TOWN",  "ULOCAL",[31,32,33])
pool(school, "RURAL", "ULOCAL",[41,42,43])

print pd.crosstab(school.ULOCAL, school.CITY)
print pd.crosstab(school.ULOCAL, school.SUBURB)
print pd.crosstab(school.ULOCAL, school.TOWN)
print pd.crosstab(school.ULOCAL, school.RURAL)

print school.MAGNET.value_counts(dropna=False)
print school.CHARTR.value_counts(dropna=False)
print school.SHARED.value_counts(dropna=False)

school["MAGNET"]=pd.to_numeric(school.MAGNET, errors="coerce")
school["CHARTR"]=pd.to_numeric(school.CHARTR, errors="coerce")
school["SHARED"]=pd.to_numeric(school.SHARED, errors="coerce")

school.ix[school.MAGNET !=1, "MAGNET"]=0
school.ix[school.CHARTR !=1, "CHARTR"]=0
school.ix[school.SHARED !=1, "SHARED"]=0

print school.MAGNET.value_counts(dropna=False)
print school.CHARTR.value_counts(dropna=False)
print school.SHARED.value_counts(dropna=False)

#School Level (ELEMENTARY, MIDDLE, HIGH, OTHER)
print school.LEVEL.value_counts(dropna=False)

school["ELEM"]=0
school["MIDDLE"]=0
school["HIGH"]=0
school["OTHER"]=0

school.ix[school.LEVEL=="1", "ELEM"]=1
school.ix[school.LEVEL=="2", "MIDDLE"]=1
school.ix[school.LEVEL=="3", "HIGH"]=1
school.ix[school.LEVEL.isin(["4","N"]),"OTHER"]=1

print pd.crosstab(school.LEVEL, school.ELEM)
print pd.crosstab(school.LEVEL, school.MIDDLE)
print pd.crosstab(school.LEVEL, school.HIGH)
print pd.crosstab(school.LEVEL, school.OTHER)

pd.crosstab(school.TITLEI, school.STITLI)

school["T1_ALL"]=0
school["T1_SOME"]=0
school["T1_NONE"]=0

school.ix[school.TITLEI=="M", "T1_ALL"]=np.nan
school.ix[school.TITLEI=="M", "T1_SOME"]=np.nan
school.ix[school.TITLEI=="M", "T1_NONE"]=np.nan

school.ix[(school.TITLEI=="1") & (school.STITLI=="1"), "T1_ALL"]=1
school.ix[(school.TITLEI=="1") & (school.STITLI=="2"), "T1_SOME"]=1
school.ix[(school.TITLEI.isin(["2","N"])), "T1_NONE"]=1

print school.T1_ALL.value_counts(dropna=False)
print school.T1_SOME.value_counts(dropna=False)
print school.T1_NONE.value_counts(dropna=False)

def trim_outliers(df, variable):
    p01=school[variable].quantile(q=0.01)
    p99=school[variable].quantile(q=0.99)
    
    school.ix[(school[variable]<p01) | (school[variable]>p99), variable]=np.nan

school.ix[school.TOTETH<0, "TOTETH"]=np.nan
school.ix[school.WHITE<0, "WHITE"]=np.nan

trim_outliers(school, "TOTETH") #clean outliers - drop top and bottom 1st percentiles
trim_outliers(school, "WHITE") #clean outliers - drop top and bottom 1st percentiles

school["PCT_MINORITY"]=((school.TOTETH-school.WHITE)/school.TOTETH)
school[["SCHNAM","WHITE","TOTETH","PCT_MINORITY"]].head(5)

#Percent Free/ Reduced Price Lunch
school.ix[school.MEMBER<0, "MEMBER"]=np.nan
school.ix[school.TOTFRL<0, "TOTFRL"]=np.nan

trim_outliers(school, "MEMBER")
trim_outliers(school, "TOTFRL")

school["PCT_FRL"]=(school.TOTFRL/ school.MEMBER)
school[["SCHNAM","MEMBER","TOTFRL","PCT_FRL"]].head(5)

school[["TOTETH","WHITE","MEMBER","TOTFRL","PCT_MINORITY","PCT_FRL"]].describe()

#Student-Teacher Ratio
school.ix[school.FTE<=0, "FTE"]=np.nan
trim_outliers(school, "FTE")

school["ST_RATIO"]=school.MEMBER/ school.FTE
trim_outliers(school, "ST_RATIO")
print school["ST_RATIO"].describe()

mathvars=["STNAM","NCESSCH","ALL_MTH00pctprof_0910"]
elavars=["STNAM","NCESSCH","ALL_RLA00pctprof_0910"]

math=pd.read_csv("data/EDfacts/2009-10/math-achievement-sch-sy2009-10.csv")[mathvars]
ela=pd.read_csv("data/EDfacts/2009-10/rla-achievement-sch-sy2009-10.csv")[elavars]

asmt=pd.merge(math,ela,on=["STNAM","NCESSCH"])
asmt.rename(columns={"ALL_MTH00pctprof_0910":"MATH_PROF","ALL_RLA00pctprof_0910":"ELA_PROF"}, inplace=True)

print math.shape, ela.shape, asmt.shape
print asmt.columns

def clean_score(raw):
    raw=str(raw)
    
    if raw in ["PS","n/a","GE50","GE40","GE30","LT50","LT40","LT30"]:
        clean=np.nan
        
    elif raw[:2] in ["GE","GT","LE","LT"]: 
        clean=float(raw[2:])

    else:
        split=str(raw).split("-")
    
        if len(split)==1:
            clean=float(split[0])
    
        elif len(split)==2:
            clean=np.mean([float(split[0]),float(split[1])])
        
        else:
            print "ERROR CHECK RANGE: "+split
    
    return clean

asmt["MATH_PROF_CLN"]=asmt.MATH_PROF.apply(clean_score)
asmt["ELA_PROF_CLN"]=asmt.ELA_PROF.apply(clean_score)

print asmt[asmt.MATH_PROF_CLN.isnull()]["MATH_PROF"].value_counts(dropna=False)
print asmt[asmt.ELA_PROF_CLN.isnull()]["ELA_PROF"].value_counts(dropna=False)

ranks=pd.DataFrame()

for state in asmt.STNAM.unique():
    state_ranks=asmt[asmt.STNAM==state].copy()
    
    state_ranks["MATH_RANK"]=asmt.MATH_PROF_CLN.rank(pct=True)
    state_ranks["ELA_RANK"]=asmt.ELA_PROF_CLN.rank(pct=True)
    
    ranks=pd.concat([ranks,state_ranks], axis=0)

ranks["in_ranks"]=1 #Flag schools that are in the EdFacts Data

print asmt.shape
print ranks.shape
print ranks.describe()

merged=pd.merge(school,ranks,on="NCESSCH",how="left")
merged.ix[merged.in_ranks.isnull(), "in_ranks"]=0 #if record is not in EdFacts data, set in_ranks=0

print merged.in_ranks.value_counts()
print school.shape, merged.shape

merged["intercept"]=1 #add intercept term for modelling

def flag_closure(yr, data, closed):
    df=pd.read_table("data/ccd/"+yr+"/"+data)[["NCESSCH","STATUS"]]
    
    df[closed]=0
    df.ix[df["STATUS"].isin([2,6]), closed]=1
  
    return df[["NCESSCH",closed]]
    
closed10=flag_closure("2010-11", "sc102a.txt", "CLOSED10")
closed11=flag_closure("2011-12", "sc111a_supp.txt", "CLOSED11")
closed12=flag_closure("2012-13", "sc122a.txt", "CLOSED12")
closed13=flag_closure("2013-14", "sc132a.txt", "CLOSED13")
closed14=flag_closure("2014-15", "Sch14pre.txt", "CLOSED14")

merged=pd.merge(merged, closed10, on="NCESSCH", how="left")
merged=pd.merge(merged, closed11, on="NCESSCH", how="left")
merged=pd.merge(merged, closed12, on="NCESSCH", how="left")
merged=pd.merge(merged, closed13, on="NCESSCH", how="left")
merged=pd.merge(merged, closed14, on="NCESSCH", how="left")

print merged.shape
print merged.columns

merged["CLOSED"]=merged[["CLOSED10","CLOSED11","CLOSED12","CLOSED13","CLOSED14"]].max(axis=1)
print merged.CLOSED.value_counts(dropna=False)

merged=merged[merged.CLOSED.notnull()]
merged.to_pickle("data/school_closure_clean.pkl")



