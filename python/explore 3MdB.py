import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

co = pymysql.connect(host='132.248.1.102', db='3MdB', user='OVN_user', passwd='oiii5007') 

res = pd.read_sql("""SELECT 
HE_2__4686A / H__1__4861A as He2, 
TOTL__3727A / H__1__4861A as O2, 
O__3__5007A / H__1__4861A as O3, 
O__1__6300A / H__1__4861A as O1, 
(S_II__6716A +  S_II__6731A )/ H__1__4861A as S2, 
OXYGEN as O,
substr(com3, 6) as age,
hbfrac,
logU_mean
FROM tab 
WHERE ref like 'BOND' 
""", 
con=co)

print(len(res))

res[0:10]

# res.age is a string, tranform it to float:
res = res.apply(pd.to_numeric, errors='ignore')

mask1 = np.abs((res['He2'] - 0.15)/0.15) < 0.2
print(mask1.sum())

res[mask1][0:10]

fig, ax = plt.subplots(figsize=(10,8))
sc = ax.scatter(np.log10(res['He2']), res['hbfrac'], c=np.log10(res['age']))
cb = fig.colorbar(sc)
ax.axvline(np.log10(0.15))
ax.set_xlabel('log HeII/Hb')
ax.set_ylabel('Fraction (1.0 is rad-bounded)');

fig, ax = plt.subplots(figsize=(10,8))
sc = ax.scatter(np.log10(res['He2']), res['hbfrac'], c=np.log10(res['age']))
ax.set_xlim((-1, -0.5))
cb = fig.colorbar(sc)
ax.axvline(np.log10(0.15))
ax.set_xlabel('log HeII/Hb')
ax.set_ylabel('Fraction (1.0 is rad-bounded)');

fig, ax = plt.subplots(figsize=(10,8))
mask_rb = res['hbfrac'] > 0.5
res_rb = res[mask_rb]
sc = ax.scatter(np.log10(res_rb['logU_mean']), res_rb['age']/1e6, c=np.log10(res_rb['He2']))
cb = fig.colorbar(sc)
cb.set_label('log(HeII/Hb)')
ax.set_xlabel('<logU>')
ax.set_ylabel('age (Myr)')

fig, ax = plt.subplots(figsize=(10,8))
mask_rb = res['hbfrac'] > 0.5
res_rb = res[mask_rb]
sc = ax.scatter(np.log10(res_rb['logU_mean']), res_rb['age']/1e6, c=np.log10(res_rb['O3']/res_rb['O2']))
cb = fig.colorbar(sc)
cb.set_label('log(OIII/OII)')
ax.set_xlabel('<logU>')
ax.set_ylabel('age (Myr)')

mask_He2 = np.log10(res['He2']) > -1.2
mask_O32 = np.log10(res_rb['O3']/res_rb['O2']) < -0.2
print(mask_He2.sum(), mask_O32.sum(), (mask_He2 & mask_O32).sum())

res2 = pd.read_sql("""SELECT 
OXYGEN as O, 
T_OXYGEN_vol_2 as TOpp, 
T_NITROGEN_vol_1 as TNp, 
logU_mean as logU 
FROM v_3MdB 
WHERE ref = 'PNe_2014' AND com6 = 1 AND hbfrac > 0.7""", 
                  con=co)

print(len(res2))

f, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
ax1.scatter(res2['TOpp'], res2['TNp']-res2['TOpp'], c=res2['O'], edgecolor='None') 
ax2.hist(res2['TNp']-res2['TOpp'], bins=50)
ax1.set_xlabel('T(O++)')
ax1.set_ylabel('T(N+) - T(O++)')
ax2.set_xlabel('T(N+) - T(O++)')
ax2.set_ylabel('Number of models');

co.close()



