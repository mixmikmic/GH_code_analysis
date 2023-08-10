import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
import seaborn as sns
get_ipython().magic('matplotlib inline')

def ImportXYZ(FileName,labelName):
    DF=pd.read_table(FileName, header=None)
    DF.columns=['Row']
    DF["Eastern"]=DF["Row"].apply(lambda x: float(x.split(" ")[0]))
    DF["Northern"]=DF["Row"].apply(lambda x: float(x.split(" ")[1]))
    DF[labelName]=DF["Row"].apply(lambda x: x.split(" ")[2])
    del DF["Row"]
    return DF

clip='~/WIFIRE/Data_Prep_Work_Flow/Clipped_Files/'

#Import Fuel
Fuel2010=ImportXYZ(clip+'Fuel2010_Escondido_UTM.xyz','Fuel2010')
Fuel2012=ImportXYZ(clip+'Fuel2012_Escondido_UTM.xyz','Fuel2012')
Fuel2014=ImportXYZ(clip+'Fuel2014_Escondido_UTM.xyz','Fuel2014')

#Combine Fuel data
Fuel=pd.merge(pd.merge(Fuel2010,Fuel2012,on=['Eastern','Northern']),Fuel2014,on=['Eastern','Northern'])

print(Fuel2010.shape)
print(Fuel2012.shape)
print(Fuel2014.shape)
print(Fuel.shape)

Fuel.head()

Hold=[]
for index, row in Fuel.iterrows():
    if row["Fuel2010"]!=row["Fuel2012"]:
        a=(row["Fuel2010"],row["Fuel2012"])
        d=1
    else:
        a=np.nan
        d=0
    if row["Fuel2012"]!=row["Fuel2014"]:
        b=(row["Fuel2012"],row["Fuel2014"])
        d+=1
    else:
        b=np.nan
        d+=0
    if d>0 :
        c=(row["Fuel2010"],row["Fuel2012"],row["Fuel2014"])
    else:
        c=np.nan
    Hold.append([a,b,c])
    
H=pd.DataFrame(Hold, columns=["Tran10-12","Tran12-14","Tran10-12-14"])

print (Fuel.shape)
Fuel=pd.concat([Fuel,H],axis=1)
print (Fuel.shape)

# What percentage did NOT change?
print ("2010 to 2012: ",Fuel[Fuel["Tran10-12"].isnull()].shape[0]/Fuel.shape[0])
print ("2012 to 2014: ",Fuel[Fuel["Tran12-14"].isnull()].shape[0]/Fuel.shape[0])
print ("2010 to 2012 to 2014: ",Fuel[Fuel["Tran10-12-14"].isnull()].shape[0]/Fuel.shape[0])

#What percentage changed back?
def RevertBack(t):
    if isinstance(t,tuple):
        if t[0]==t[2]:
            return t[0]
        else:
            return np.nan
    else:
        return np.nan
Fuel["Reverted"]=Fuel['Tran10-12-14'].apply(lambda x: RevertBack(x))
print("Changed from 2010 to 2012, then Changed Back 2014:",Fuel[Fuel["Reverted"].isnull()==False].shape[0]/Fuel.shape[0])

Revert=Fuel.groupby(["Reverted","Fuel2012"])["Eastern"].count().reset_index()
Revert.columns=["Fuel2010&2014","Fuel2012","Count"]
Revert.head()

#Import Fuel "decoder ring"
DecodeFuel=pd.read_csv('~/WIFIRE/Data_labels/fuel_labels.csv')
DecodeFuel.columns=["filename",'id','fuelLabel']
DecodeFuel["id"]=DecodeFuel["id"].apply(lambda x: str(x))
DecodeFuel

Revert=pd.merge(pd.merge(Revert,DecodeFuel[["id","fuelLabel"]],left_on="Fuel2010&2014",right_on="id",how='inner'),DecodeFuel[["id","fuelLabel"]],left_on='Fuel2012',right_on='id',how='inner')
Revert.columns=["Fuel2010&2014","Fuel2012","Count","id_x","Label10&14","id_y","Label12"]
Revert[["Fuel2010&2014","Fuel2012","Count","Label10&14","Label12"]].to_csv('RevertedFuel.csv',index=False)
Revert.head()

print "2010-2012 Tiles That Stayed The Same:", len(Fuel.loc[Fuel['Fuel2010'] == Fuel['Fuel2012']])
print "2012-2014 Tiles That Stayed The Same:",len(Fuel.loc[Fuel['Fuel2012'] == Fuel['Fuel2014']])
print "2010-2012 Tiles That Changed:", len(Fuel.loc[Fuel['Fuel2010'] != Fuel['Fuel2012']])
print "2012-2014 Tiles That Changed:", len(Fuel.loc[Fuel['Fuel2012'] != Fuel['Fuel2014']])
fueldiff10_12 = Fuel.loc[Fuel['Fuel2010'] != Fuel['Fuel2012']]
fueldiff12_14 = Fuel.loc[Fuel['Fuel2012'] != Fuel['Fuel2014']]

ls_labels = Fuel.loc[Fuel['Fuel2010'] != Fuel['Fuel2012']].Fuel2010.unique()
print ls_labels

sum_diff10_12 = fueldiff10_12.groupby(['Fuel2010', 'Fuel2012']).size().rename('Count_Pixel').reset_index()
sum_diff12_14 = fueldiff12_14.groupby(['Fuel2012', 'Fuel2014']).size().rename('Count_Pixel').reset_index() 

Label_2010 = DecodeFuel[['id','fuelLabel']]
Label_2012 = DecodeFuel[['id','fuelLabel']]
Label_2014 = DecodeFuel[['id','fuelLabel']]
Label_2010.columns = ['Fuel2010','FuelLabel_2010']
Label_2012.columns = ['Fuel2012','FuelLabel_2012']
Label_2014.columns = ['Fuel2014','FuelLabel_2014']

DecodeFuel[['id','fuelLabel']]

crosstab10_12 = pd.crosstab(diff10_12.Fuel2010, diff10_12.Fuel2012, values=diff10_12.Count_Pixel, 
            aggfunc=np.sum)
ax = sns.heatmap(crosstab10_12,  linewidths=2, cmap="YlGnBu")

labeled_10_12 = sum_diff10_12.merge(Label_2010, on = 'Fuel2010')
labeled_10_12 = labeled_10_12.merge(Label_2012, on = 'Fuel2012')
labeled_10_12.head()

diff_top_labels = ['Short Grass','Timber Grass', 'Brush', 'Urban']

fig, axes = plt.subplots(nrows=2, ncols=2)
plt.subplots_adjust(wspace=0.9, hspace=1);
labeled_10_12[labeled_10_12['FuelLabel_2010'] == 'Short Grass'].plot(ax=axes[0,0],
x="FuelLabel_2012", y="Count_Pixel", kind='bar')
labeled_10_12[labeled_10_12['FuelLabel_2010'] == 'Timber Grass'].plot(ax=axes[0,1],
x="FuelLabel_2012", y="Count_Pixel", kind='bar')
labeled_10_12[labeled_10_12['FuelLabel_2010'] == 'Brush'].plot(ax=axes[1,0],
x="FuelLabel_2012", y="Count_Pixel", kind='bar')
labeled_10_12[labeled_10_12['FuelLabel_2010'] == 'Urban'].plot(ax=axes[1,1],
x="FuelLabel_2012", y="Count_Pixel", kind='bar')

crosstab12_14 = pd.crosstab(sum_diff12_14.Fuel2012, sum_diff12_14.Fuel2014, values=sum_diff12_14.Count_Pixel, 
            aggfunc=np.sum)
ax = sns.heatmap(crosstab12_14,  linewidths=2, cmap="YlGnBu")

labeled_12_14 = sum_diff12_14.merge(Label_2012, on = 'Fuel2012')
labeled_12_14 = labeled_12_14.merge(Label_2014, on = 'Fuel2014')
labeled_12_14.head()

diff_top_labels = ['Short Grass','Timber Grass', 'Brush', 'Urban']

fig, axes = plt.subplots(nrows=2, ncols=2)
plt.subplots_adjust(wspace=0.9, hspace=1);

labeled_12_14[labeled_12_14['FuelLabel_2012'] == 'Short Grass'].plot(ax=axes[0,0],
x="FuelLabel_2014", y="Count_Pixel", kind='bar')
labeled_12_14[labeled_12_14['FuelLabel_2012'] == 'Timber Grass'].plot(ax=axes[0,1],
x="FuelLabel_2014", y="Count_Pixel", kind='bar')
labeled_12_14[labeled_12_14['FuelLabel_2012'] == 'Brush'].plot(ax=axes[1,0],
x="FuelLabel_2014", y="Count_Pixel", kind='bar')
labeled_12_14[labeled_12_14['FuelLabel_2012'] == 'Urban'].plot(ax=axes[1,1],
x="FuelLabel_2014", y="Count_Pixel", kind='bar')



