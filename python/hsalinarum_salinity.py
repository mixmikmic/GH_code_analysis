import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from datetime import timedelta

get_ipython().magic('matplotlib inline')

data_dir = "data/hsalinarum/beer_et_al_2014/"
plates = os.listdir(data_dir)
plates.remove('temp')
plates = [p for p in plates if not "." in p]
plates

combined = pd.DataFrame()
key = pd.DataFrame()
data = pd.DataFrame()

for p in plates:
    files = os.listdir(os.path.join(data_dir,p))
    
    database = pd.read_csv(os.path.join(data_dir,p,"DatabaseKB"+p[:-2]+".txt"))
    if database.shape[1] == 1:
        database = pd.read_csv(os.path.join(data_dir,p,"DatabaseKB"+p[:-2]+".txt"),sep="\t")
    database['plate'] = p
    
    result = pd.read_csv(os.path.join(data_dir,p,"ResultsKB"+p[:-2]+".csv"),index_col=False)
    
    # if there is an unnamed index before time column, delete it
    if result.columns[0] != 'Time':
        del result[result.columns[0]]
        
    result.columns = ['Time'] + result.columns[1:].str.extract("Well[ .]?([0-9]*)").tolist()
        
    # convert times to hour values using the timedelta fxn
    tdelt = result.Time.str.split(":").apply(lambda x: timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2])))
    tdelt = tdelt - tdelt.values[0]
    tdelt = tdelt/np.timedelta64(1,'h') # convert to hours
    tdelt = tdelt.round(1)
    result['Time'] = tdelt
    
    # merge results and database
    database['Well'] = result.columns[1:]
    merged = pd.merge(database,result.iloc[:,1:].T,left_on='Well',right_index=True)
    merged.columns = merged.columns[:database.shape[1]].tolist() + result.Time.tolist()
    
    merged.to_csv(os.path.join(data_dir,p+'_merged.csv'),index=False)
    result.to_csv(os.path.join(data_dir,p+'_data.csv'),index=False)
    database.to_csv(os.path.join(data_dir,p+'_design.csv'),index=False)
    
    if combined.shape[0]==0:
        combined = merged
        key = database
        data = result
    else:
        combined = combined.append(merged)
        key = key.append(database)
        
        result.index = range(data.shape[0],data.shape[0]+result.shape[0])
        data = data.append(result)
    print p,data.shape, key.shape

# this is the first column that isn't a data measurement
time_ind = combined.columns.tolist().index('Arg.Concentration')

select = ['Total.Concentration',"NaCl.Concentration","MgSO4.Concentration","KCl.Concentration"]

combined = combined[~combined[select].isnull().any(1)]

combined.iloc[:,:time_ind] = np.log2(combined.iloc[:,:time_ind])

# combined.iloc[:,:time_ind] = (combined.iloc[:,:time_ind].values.T - combined.iloc[:,0].values.T).T

g = combined.groupby(['Total.Concentration',"NaCl.Concentration","MgSO4.Concentration","KCl.Concentration"])

len(g.groups.keys())

keys = g.groups.keys()
keys.sort()
keys

concs = np.unique([k[0] for k in keys])
concs.sort()
concs

composite = g.apply(lambda x: x.loc[x.iloc[:,0]<-1,:].iloc[:,26:time_ind].mean(0))
composite = pd.DataFrame((composite.values.T-composite.values[:,0]).T,index=composite.index,columns=composite.columns)

time = composite.columns.astype(float)
diff = time[1:] - time[:-1]
ind = np.where(diff-.1 < 1e-6)[0]
ind

time

for i in ind:
    select = np.where(composite.iloc[:,i+1].isnull())[0]
    composite.iloc[select,i+1] = composite.iloc[select,i]

composite = composite.drop(composite.columns[ind],axis=1)

composite = composite.loc[:,composite.columns <= 95]

temp = np.array(composite.index.tolist())
temp[:,1] = temp[:,1].T/(np.sum(temp[:,1:],1))
temp[:,2] = temp[:,2]/(np.sum(temp[:,2:],1))
temp = temp.round(3)
temp = temp[:,:3]
temp

composite.index = pd.MultiIndex.from_tuples([tuple(t) for t in temp],names=['total','Na/(Na+Mg+K)','Mg/(Mg+K)'])

composite.to_csv("data/hsalinarum/beer_et_al_2014/composite.csv",index=True)

totals = [2.633,3.072,3.51,3.949,4.388]
nacls = [.586,.779,0.975]
labeled={}
r = .4
plt.figure(figsize=(30,10))
select = composite.columns<50
for i in composite.index:
    total,nacl,mg = i
    
    plt.subplot(1,5,totals.index(total)+1)
    plt.title(str(total),fontsize=30)
    
    if total == 2.633 and not nacl in labeled:
        l = str(nacl)+' NaCl'
        labeled[nacl] = None
        plt.plot(composite.columns[select],composite.loc[i,select].values,c=cmap(r+(1.-r)*nacls.index(nacl)/len(nacls)),label=l)
    else:
        plt.plot(composite.columns[select],composite.loc[i,select].values,c=cmap(r+(1.-r)*nacls.index(nacl)/len(nacls)),)
    plt.ylim(-.3,2.4)
    
    
    
plt.subplot(151)
plt.ylabel("log(OD)",fontsize=25)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("time (h)",fontsize=25)


plt.subplot(1,5,1)
plt.legend(loc="best",fontsize=20)

plt.savefig("figures/hsalinarum_sal/nacl_by_total.png",bbox_inches="tight")

composite.head()

def plot_2d(d1,d2,ylim=None):
    dims = {'total':[2.633,3.072,3.51,3.949,4.388],
            'Na/(Na+Mg+K)':[.586,.779,0.975],
            'Mg/(Mg+K)':[0,.25,.5,.75,1]}
    
    cdim = dims.keys()
    cdim.remove(d1)
    cdim.remove(d2)
    cdim = cdim[0]
    
    select = composite.columns<50
    r = .4
    
    ncols = len(dims[d1])
    nrows = len(dims[d2])
    
    order = [d1,d2,cdim]
    
    for v1 in dims[d1]:
        for v2 in dims[d2]:
            for vc in dims[cdim]:
                pos = dims[d1].index(v1)+len(dims[d1])*dims[d2].index(v2)+1
                plt.subplot(nrows,ncols,pos)
                
                #plt.title("%s=%s, %s=%s"%(d1,str(v1),d2,str(v2)))
                if (pos - 1)%ncols == 0:
                    plt.ylabel("%s=%.2lf"%(d2,v2),rotation=0,fontsize=30)
                if (pos - 1)/ncols == 0:
                    plt.title("%s=%.2lf"%(d1,v1),rotation=0,fontsize=30)
                
                i = v1,v2,vc
                i = [i[order.index(z)] for z in ['total','Na/(Na+Mg+K)','Mg/(Mg+K)']]
                i = tuple(i)
                
                plt.plot(composite.columns[select],composite.loc[i,select].values,c=cmap(r+(1.-r)*dims[cdim].index(vc)/len(dims[cdim])),)            
    
                if not ylim is None:
                    plt.ylim(ylim)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

plt.figure(figsize=(40,20))
plot_2d('total','Na/(Na+Mg+K)',(-.3,2.3))
plt.tight_layout()

plt.figure(figsize=(30,20))
plot_2d('total','Mg/(Mg+K)',(-.3,2.3))
plt.savefig("figures/hsalinarum_sal/nacl_2d.png",bbox_inches='tight')

plt.figure(figsize=(20,14))

cmap=plt.get_cmap('Blues')
min_conc = 1.5
max_conc= 4.3
offset = .1

for k in keys:
    i = concs.tolist().index(k[0]) + 1
    plt.subplot(2,3,i)
    
    temp = g.get_group(k)
    time = temp.columns[:time_ind]
    od = temp.iloc[:,:time_ind].mean(0)
    
    # od = od.loc[:,time<50]
    od = od[time<50]
    time = time[time<50]
    
    #od = od.loc[od.iloc[:,0]<-1,:]
#     od = od[od.iloc[0]<-1]
    
#     time = time[~od.isnull()]
#     od = od[~od.isnull()]
    
    plt.plot(time,od.values.T,'-',c=cmap((k[1]-min_conc+offset)/(max_conc-min_conc+2*offset)),label='%.2lf M NaCl'%k[1],lw=2,alpha=.5)
    
for c in concs:
    i = concs.tolist().index(c) + 1
    plt.subplot(2,3,i)
    plt.title("total salt: %.2lf"%c,fontsize=30)
    
#     plt.ylim(-4,-.5)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    if i == 1 or i == 4:
        plt.ylabel("log(OD)",fontsize=30)
    if i>=3:
        plt.xlabel("time (h)",fontsize=30)
    
plt.tight_layout()

plt.savefig("hsalinarum_salinity_bySalt_nacl.png",bbox_inches="tight")

plt.figure(figsize=(20,6))
for k in keys:
    if not k[0] == 3.9492:
        continue
    
    temp = g.get_group(k)
    time = temp.columns[:time_ind]
    od = temp.iloc[:,:time_ind].mean(0)
    
    od = od[time>5]
    time = time[time>5]
    
    od = od[time<50]
    time = time[time<50]
    
    time = time[~od.isnull()]
    od = od[~od.isnull()]
    
    od = od-od.values[0]
    
    plt.subplot(121)
    plt.plot(time,od,'-',c=cmap((k[1]-min_conc+offset)/(max_conc-min_conc+2*offset)),label='%.2lf M NaCl'%k[1],lw=2)
    
    plt.subplot(122)
    plt.plot(time,od,'-',c=cmap((k[2]-0+.5)/(1.7-0+2*.5)),label='%.2lf M Mg'%k[1],lw=2)
    
plt.subplot(121)
plt.title("NaCl",fontsize=30)
plt.ylim(-.1,2.4)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("log(OD)",fontsize=30)
plt.xlabel("time (h)",fontsize=30)

plt.subplot(122)
plt.title("Mg/KCl",fontsize=30)
plt.ylim(-.1,2.4)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel("time (h)",fontsize=30)

plt.savefig("hsalinarum_salinity_single_nacl_mgkcl.png",bbox_inches="tight")

plt.figure(figsize=(20,10))

cmap=plt.get_cmap('Blues')
min_conc = 0
max_conc= .05

for k in keys:
    i = concs.tolist().index(k[0]) + 1
    plt.subplot(2,3,i)
    
    temp = g.get_group(k)
    time = temp.columns[:time_ind]
    od = temp.iloc[:,:time_ind].mean(0)
    plt.plot(time,od,c=cmap((k[2]-min_conc)/(max_conc-min_conc)),label='%.2lf M NaCl'%k[1])
    
for c in concs:
    i = concs.tolist().index(c) + 1
    plt.subplot(2,3,i)
    plt.title("total salt: %.2lf"%c,fontsize=30)
    
    plt.ylim(-.5,5)

combined.to_csv("data/hsalinarum/beer_et_al_2014/combined.csv",index=False)

effect = combined[['KCl.Concentration','MgSO4.Concentration','NaCl.Concentration','Total.Concentration']]

effect.columns = effect.columns.str.replace(".","_")
effect.values[:,:3] = (effect.values[:,:3]/effect.Total_Concentration.values[:,None])

effect.loc[:,'KCl_Concentration'] = effect.KCl_Concentration.factorize()[0]
effect.loc[:,'MgSO4_Concentration'] = effect.MgSO4_Concentration.factorize()[0]
effect.loc[:,'NaCl_Concentration'] = effect.NaCl_Concentration.factorize()[0]
effect.loc[:,'Total_Concentration'] = effect.Total_Concentration.factorize()[0]

effect



import patsy

m = patsy.dmatrix('0+C(KCl_Concentration)+C(MgSO4_Concentration)+C(NaCl_Concentration)+C(Total_Concentration)',effect)
m.shape

np.array(m)

time_ind = 563
combined.columns[:time_ind]

tidy = pd.melt(combined,id_vars=combined.columns[time_ind:].tolist(),
        value_vars=combined.columns[:time_ind].tolist(),
        var_name="time",value_name='od')

tidy.to_csv("data/hsalinarum/beer_et_al_2014/tidy.csv",index=False)

temp = g.get_group(keys[0])
time = temp.columns[:time_ind]
od = temp.iloc[:,:time_ind]
plt.plot(time,od.T)

plt.figure(figsize=(60,20))

ylim = (np.inf,-np.inf)

for i,k in enumerate(keys):
    
    plt.subplot(5,15,i+1)
    
    temp = g.get_group(k)
    time = temp.columns[:time_ind]
    od = temp.iloc[:,:time_ind]
    plt.plot(time,od.T)
    
    s = ','.join("%.2lf" % v for v in k[1:])
    
    plt.title(s,fontsize=20)
    
    ylim = (min(od.min().min(),ylim[0]),max(od.max().max(),ylim[1]))
    
    
# standardize ylim for all plots
for i in range(len(keys)):
    plt.subplot(5,15,i+1)
    plt.ylim(ylim)
    
plt.savefig("hsalinarum_salinity_raw_data.pdf",bbox_inches="tight")



