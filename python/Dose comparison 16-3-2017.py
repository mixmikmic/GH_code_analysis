get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='ticks')

def d1_id(df):    # helper function to get the index when Mass density = 1.0
    return df[df['Mass density']==1.0].index.tolist()[0]
#d1_id(EGS)

def n_r(df, raw_col = 'Raw_data_col', norm_col = 'Normalised_data_col'): # norm and return, pass the dataframe, column to normalise and drop, and the norm col name
    if raw_col in df.columns:
        df[norm_col] = df[raw_col]/df[raw_col].iloc[d1_id(df)]
        df = df.drop(raw_col, 1) 
        return df

# EGS = n_r(EGS, raw_col = 'EGS_Dose', norm_col = 'EGS_Dose_norm')

ECLIPSE = pd.read_csv("Eclipse_data_19_10.csv")  # index_col=[0]
ECLIPSE = n_r(ECLIPSE, raw_col = 'AAA (Gy) point', norm_col = 'AAA')
ECLIPSE = n_r(ECLIPSE, raw_col = 'AXB Dw (Gy) point', norm_col = 'AXB Dw')
ECLIPSE = n_r(ECLIPSE, raw_col = 'AXB Dm (Gy) point', norm_col = 'AXB Dm')
df = ECLIPSE
#df

df=df.set_index(['Mass density'])

metrics_ = [ 'AAA', 'AXB Dw', 'AXB Dm', 'CCC', 'EGS']
df['Mean'] = df.loc[:,metrics_].mean(axis=1)   # apply mean by row
df['Max'] = df.loc[:,metrics_].max(axis=1)   # apply mean by row
df['Min'] = df.loc[:,metrics_].min(axis=1)   # apply mean by row

df['Max % error'] = df.loc[:,'Max']   # apply mean by row

df['Max % error'] = 100.0*(df['Max'] - df['Min'])/df['Min']   # apply mean by row

df2 = df[['VOI Material', 'Proton Stp PWR','CT number', 'AAA', 'AXB Dw', 'AXB Dm', 'Max', 'Mean', 'Max % error']]   # 'CCC', 'EGS', rearrange, , 'Elect density', 'Proton Stp PWR', 'Min', 'Max',  
df2 = df2.rename(columns={'Proton Stp PWR': 'Stp PWR'})
df2

#df2.to_csv('ALL_DATA_8_3_2017.csv', index=False)  # write to file

df2['VOI Material'].values

EGS = pd.read_csv("EGS_data_19_10.csv", index_col=[0])  # 

width = 8
height = 8
annotation_height = 0.895
annotation_x_offset = -.035
plt.figure(figsize=(width, height))  # width, height

if False:
    df = df2
else:    # plot only biological?
    df = df2.iloc[[0,1,2,3,5,6,13]]

plt.plot(df[['AAA']], c='g', marker='o', label='AAA Dw', ls='--')
plt.plot(df[['AXB Dw']], c='r', marker='o', label='AXB Dw')
plt.plot(df[['AXB Dm']], c='b', marker='o', label='AXB Dm')
plt.plot(EGS['EGS_Dose_norm'], c='m', marker='o', label='EGS Dm', ls='--')

for i, txt in enumerate(df2['VOI Material'].values):
    if txt in ['Air', 'Lung', 'Adipose tissue', 'Muscle skeletal','Bone', 'Aluminium', 'Water']:   # 'Adipose tissue',   'PVC', 
   # if txt in df2['VOI Material'].values:    
        plt.annotate(txt, (df2['Max'].index.values[i]+annotation_x_offset, annotation_height), rotation=90)  # df2['Max'].values[i]

axes = plt.gca()
axes.set_xlim([-0.05,2.5]) # min and max
axes.set_ylim([0.8,1.15])

plt.xlabel(r'Mass density ($\mathbf{g/cm^{3}}$)')
plt.ylabel('Relative dose (A.U)')
plt.title('Dose relative to water')

plt.tick_params(top='off', right='off')      # ticks along the bottom edge are off

plt.legend(shadow=True, fontsize='small', loc='top left') ,
#plt.grid(True)
#plt.savefig('Fig1.png' , dpi=500);#, format='pdf'
plt.show();

df2.iloc[[0,1,2,3,5,6,13]]



