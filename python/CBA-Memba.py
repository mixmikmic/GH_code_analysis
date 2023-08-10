import prim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

inputs = pd.read_csv("../Results_DMU/Murrumbala/scenario_values.csv")

inputs.Climate_change_scenario.replace(2,0,inplace=True)

cost = pd.read_csv("../Results_DMU/Memba/COST_d7_v2.csv")
npv = pd.read_csv("../Results_DMU/Memba/NPV_d7_v2.csv")

reduced_risk = pd.read_csv("../Results_DMU/Memba/BEN_ind_d7_v2.csv",dtype='float64')
avoided_repair = pd.read_csv("../Results_DMU/Memba/BEN_direct_d7_v2.csv",dtype='float64')
ruc_decrease = pd.read_csv("../Results_DMU/Memba/BEN_add_d7_v2.csv",dtype='float64')
isolated_trips = pd.read_csv("../Results_DMU/Memba/BEN_isol_d7_v2.csv",dtype='float64')
maintenance_sav = pd.read_csv("../Results_DMU/Memba/BEN_sav_d7_v2.csv",dtype='float64')

benef = reduced_risk+avoided_repair+ruc_decrease+maintenance_sav

import seaborn as sns
sns.set_context("notebook",rc={"font.size": 18})
sns.set_style("whitegrid")

reduced_risk.sample(3)

def savefig(path, **kwargs):
    #Saves in both png and pdf
    
    plt.tight_layout()
    
    path = path.replace(".png","")
    path = path.replace(".pdf","")

    plt.savefig(path+".png", )
    plt.savefig(path+".pdf", )

new_names = dict({'Paving link#1':'Pave1', 'Paving link#2':'Pave2', 'Improve drainange all':'drainage',
       'Gravel link #1 and #2':'gravel12', 'Gravel alternatives routes':'gravelothers'})

new_names2 = dict({'Paving link#1':'Inv. 1', 'Paving link#2':'Inv. 2', 'Improve drainange all':'Inv. 3',
       'Gravel link #1 and #2':'Inv. 4', 'Gravel alternatives routes':'Inv. 5'})

figfolder = "Memba/"

fig = plt.figure(figsize=(10,10))
plt.subplot(221)
sns.boxplot(10**(-6)*reduced_risk.rename(columns=new_names2),width=0.5)
plt.ylabel("Million USD")
plt.title("Reduced flood risk")
plt.subplot(222)
sns.boxplot(10**(-6)*avoided_repair.rename(columns=new_names2),width=0.5)
plt.title("Avoided repair")
plt.subplot(223)
sns.boxplot(10**(-6)*ruc_decrease.rename(columns=new_names2),width=0.5)
plt.title("RUC decrease")
plt.ylabel("Million USD")
plt.subplot(224)
sns.boxplot(10**(-6)*maintenance_sav.rename(columns=new_names2),width=0.5)
plt.ylabel("Million USD")
plt.title("Difference in maintenance costs")

savefig(figfolder+"boxplot_benefit_break_down")

def anova_table(varin,data,experiments_cols):
    formula = varin+" ~ " + "+".join(experiments_cols)
    olsmodel=ols(formula,data=data).fit()
    table=anova_lm(olsmodel)
    table['sum_sq_pc']=table['sum_sq']/table['sum_sq'].sum()
    table=table.sort(['sum_sq'],ascending=False)
    return table['sum_sq_pc']

reduced_risk.columns

anova_reduced_risk=pd.DataFrame(index=reduced_risk.rename(columns=new_names).columns,columns=inputs.columns)
for i in reduced_risk.rename(columns=new_names).columns:
    anova_reduced_risk.loc[i,:]=anova_table(i,pd.concat([inputs,reduced_risk],axis=1).rename(columns=new_names),inputs.columns)

anova_reduced_risk

anova_avoided_repair=pd.DataFrame(index=avoided_repair.rename(columns=new_names).columns,columns=inputs.columns)
for i in avoided_repair.rename(columns=new_names).columns:
    anova_avoided_repair.loc[i,:]=anova_table(i,pd.concat([inputs,avoided_repair],axis=1).rename(columns=new_names),inputs.columns)

anova_avoided_repair

anova_ruc_decrease=pd.DataFrame(index=ruc_decrease.rename(columns=new_names).columns,columns=inputs.columns)
for i in ruc_decrease.rename(columns=new_names).columns:
    anova_ruc_decrease.loc[i,:]=anova_table(i,pd.concat([inputs,ruc_decrease],axis=1).rename(columns=new_names),inputs.columns)

anova_ruc_decrease

anova_total=pd.DataFrame(index=benef.rename(columns=new_names).columns,columns=inputs.columns)
for i in benef.rename(columns=new_names).columns:
    anova_total.loc[i,:]=anova_table(i,pd.concat([inputs,benef],axis=1).rename(columns=new_names),inputs.columns)

anova_total

cost.name="cost"
reduced_risk.name="reduced_risk"
avoided_repair.name="avoided_repair"
ruc_decrease.name="ruc_decrease"
benef.name="benef"
isolated_trips.name="isolated_trips"
maintenance_sav.name='maintenance_sav'

a = cost.unstack()
b = benef.unstack()
c = isolated_trips.unstack()

a.index.names = ["intervention","scenario"]
b.index.names = ["intervention","scenario"]
c.index.names = ["intervention","scenario"]

a.index.difference(b.index)

m = pd.concat([pd.DataFrame(a,columns=["cost"]),pd.DataFrame(b,columns=["benefits"]),pd.DataFrame(c,columns=["isolated_trips"])],axis=1)

inputs.index.name="scenario"

m = m.reset_index().merge(inputs.reset_index()).set_index(["intervention","scenario"])

m.columns

m['npv']=m.benefits-m.cost
m['saved_trip_per_dollar'] = m.isolated_trips/m.cost
m['benefit_cost_ratio'] = m.benefits/m.cost

df3 = pd.DataFrame()

for i in ['Gravel alternatives routes', 'Gravel link #1 and #2', 'Improve drainange all', 'Paving link#1', 'Paving link#2']:

    formula="npv ~ Repair_time + Discount_Rate + Traffic_growth + Flood_duration + Agriculture_elas + Climate_change_scenario"
    olsmodel=ols(formula,data=m.loc[i]).fit()
    table=anova_lm(olsmodel)
    df3[i]=table['sum_sq']/table['sum_sq'].sum()
    
df3

df = pd.DataFrame()

for i in ['Gravel alternatives routes', 'Gravel link #1 and #2', 'Improve drainange all', 'Paving link#1', 'Paving link#2']:

    formula="benefits ~ Repair_time + Discount_Rate + Traffic_growth + Flood_duration + Agriculture_elas + Climate_change_scenario"
    olsmodel=ols(formula,data=m.loc[i]).fit()
    table=anova_lm(olsmodel)
    df[i]=table['sum_sq']/table['sum_sq'].sum()

df

ax=df.T.plot(kind='bar')
ax.legend(bbox_to_anchor=(1.7, 1.))

df2 = pd.DataFrame()

for i in ['Gravel alternatives routes', 'Gravel link #1 and #2', 'Improve drainange all', 'Paving link#1', 'Paving link#2']:

    formula="cost ~ Repair_time + Discount_Rate + Traffic_growth + Flood_duration + Agriculture_elas + Climate_change_scenario"
    olsmodel=ols(formula,data=m.loc[i]).fit()
    table=anova_lm(olsmodel)
    df2[i]=table['sum_sq']/table['sum_sq'].sum()

ax=df2.T.plot(kind='bar')
ax.legend(bbox_to_anchor=(1.7, 1.))

m.unstack('intervention').swaplevel(i=1,j=0,axis=1).sample(10)

ax=plt.figure(figsize=(10,5))
plt.subplot(121)
ax=sns.boxplot(x='intervention',y='benefit_cost_ratio',data=m.rename(index=new_names2).reset_index(),width=0.5)
plt.subplot(122)
ax=sns.boxplot(x='intervention',y='saved_trip_per_dollar',data=m.rename(index=new_names2).reset_index(),width=0.5)

savefig(figfolder+"cba_results")

cc = dict({0:"CC decreases\nprecipitations",1:"Current\nclimate", 3:"Medium\nincrease in\nprecipitations", 4:"High increase\nin precipiations"})

sub=m.rename(index=new_names).reset_index()
sub.Climate_change_scenario.replace(cc,inplace=True)
sns.boxplot(x='Climate_change_scenario',y='benefit_cost_ratio',data=sub[sub.intervention=='drainage'],width=0.5)
plt.title("Improve drainage all")
#savefig("benefit_cost_ratio_drainage")

sub=m.rename(index=new_names).reset_index()
sub.Climate_change_scenario.replace(cc,inplace=True)
sns.boxplot(x='Climate_change_scenario',y='benefit_cost_ratio',data=sub[sub.intervention=='Pave1'],width=0.5)
plt.title("Pave link #1")
#savefig("benefit_cost_ratio_pave_1")

regret = m.npv.unstack('intervention').copy()
regret = regret.rename(columns=new_names2).sort(axis=1)

r = regret.add(-regret.max(axis=1),axis=0)

regret.add(-regret.max(axis=1),axis=0).min()

fig = plt.figure(figsize=(5,5))
sns.boxplot(-10**(-6)*regret.add(-regret.max(axis=1),axis=0),width=0.5)
plt.ylabel("Regret (Million USD)")
savefig(figfolder+"regret")

rr=r==0
rr.sum(axis=0)/len(rr)

import prim

forprim=m.unstack('intervention').swaplevel(i=1,j=0,axis=1).rename(columns=new_names)

sum(scenarofinterest.npv>0)/2000

forprim.columns

plt.scatter(scenarofinterest.Repair_time,scenarofinterest.npv)

scenarofinterest = forprim.Pave1
p = prim.Prim(scenarofinterest[inputs.columns], scenarofinterest.npv>0, threshold=0.5, threshold_type=">")

box = p.find_box()
box.show_tradeoff()

plt.show()

box.select(8)
print(box)

box.show_details();

scenarofinterest = forprim.Pave1
p = prim.Prim(scenarofinterest[inputs.columns], forprim.Pave1.npv>forprim.drainage.npv, threshold=0.5, threshold_type=">")

box = p.find_box()
box.show_tradeoff()

plt.show()

box.select(5)
print(box)

box.show_details();

sum(forprim.Pave1.npv>forprim.drainage.npv)/2000

scenarofinterest = forprim.Pave1
p = prim.Prim(scenarofinterest[inputs.columns], forprim.Pave1.npv<forprim.drainage.npv, threshold=0.5, threshold_type=">")

box = p.find_box()
box.show_tradeoff()

plt.show()

box.select(15)
print(box)

box.show_details();



