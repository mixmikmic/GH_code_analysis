## spatial weights with pysal
import pysal as ps
import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import statsmodels.api as st
import patsy
import numpy as np
import glob
import os
drop=os.getenv('DROPBOX_LOC')

from shapely.geometry import Point
from fiona.crs import from_epsg
import geopandas as gpd
from pyproj import Proj

import matplotlib.pyplot as plt
import matplotlib.cm as cm, matplotlib.font_manager as fm
import seaborn as sns
#sns.set(style="darkgrid")
plt.style.use('fivethirtyeight')
get_ipython().magic('pylab inline')

files=glob.glob(os.path.join(drop,'Documents/Data/Housing/redfin','redfin_2017-06*'))

## Load and drop dupes
keeps=[]
for f in files:
    temp=pd.read_csv(f,parse_dates=['SOLD DATE'])
    keeps.append(temp)
redfin=pd.concat(keeps)
redfin=redfin.drop_duplicates('MLS#')

redfin=redfin.rename(columns=lambda x: x.replace(' ','_'))
print redfin.shape
redfin=redfin[redfin.SOLD_DATE.notnull()]
print redfin.shape
redfin['year']=redfin.SOLD_DATE.map(lambda x: x.year)
redfin['month']=redfin.SOLD_DATE.map(lambda x: x.month)
redfin['quarter']=redfin.SOLD_DATE.map(lambda x: x.quarter).astype(str)
redfin['bldg_age_at_sale']=redfin.year-redfin.YEAR_BUILT
redfin['sqft_100']=redfin.SQUARE_FEET/100
redfin['sqft_100_2']=redfin['sqft_100']*redfin['sqft_100']
redfin.LOCATION=redfin.LOCATION.replace({'Montclaire':'Montclair','Montclair Distri':'Montclair',
                                         'Crocker Hghlands':'Crocker Highlands','Glenview (Uppr)':'Upper Glenview',
                                        'Grand Lake/ Rose':'Grand Lake','TemescalRRidge':'Temescal Ridge',
                                        'Oakmore Upper':'Upper Oakmore','Fruitvale Area':'Fruitvale',
                                         'Fruitvale District':'Fruitvale'}).str.title()




redfin[redfin.LONGITUDE.notnull()]
redfin.LOCATION=redfin.LOCATION.str.replace('Dist.$','District').str.replace(' |\-','_').str.replace('\.|/|\-','')

redfin['LOCATION_recode']=redfin.LOCATION.str.extract('(Glenview|Piedmont|Oakmore)').fillna('Other')

## Turn to geodataframe

redfin['geometry'] = redfin.apply(lambda x: Point(x.LONGITUDE,x.LATITUDE),axis=1)
redfin = gpd.GeoDataFrame(redfin).set_geometry(col='geometry',crs=from_epsg(4326))

## define variable groups

coords = ['LONGITUDE','LATITUDE']
xvars=['sqft_100','bldg_age_at_sale','year','BATHS','BEDS','LOCATION_recode']
yvar=['PRICE']

subset = redfin.ix[:,yvar+xvars+coords].dropna().drop_duplicates(coords).reset_index()
print subset.shape
subset.head()

w = ps.weights.KNN(subset.loc[:,coords                              ].values, k=3)
w.transform = 'R'

w.s2array.shape

glenview = redfin[redfin.LOCATION=='Glenview']
glenview.groupby(['year']).PRICE.quantile([.25,.5,.75]).unstack(1).plot(cmap='viridis')

ax = sns.kdeplot(glenview.LONGITUDE,glenview.LATITUDE, shade=True, cmap='viridis');

## get dummies for string vars--a bit silly since it is really just the LOCATION variable.

vartypes=subset[xvars].dtypes=='object'
vartypes=vartypes[vartypes==True].index
vartypes

dummies = pd.get_dummies(subset[vartypes])
print dummies.shape
dummies.head()

subset=subset.merge(dummies,left_index=True,right_index=True)
subset.head()

## Get a list of all enumerated dummy var names, with LOCATION prepended
fulldummylist =[] 
for v in vartypes:
    for c in subset.columns:
        #print c
        if v in c and v!=c:
            
            fulldummylist.append(c)

## remove glenview so dummies are not perfectly collinear
fulldummylist.remove('LOCATION_recode_Glenview')

xvars

xvars_num=list(xvars)
for rem in vartypes:
    xvars_num.remove(rem)

## Version treating neighborhoods as contrasts relative to Glenview as base category

sm_ols_contrast_a = sm.ols("np.log(PRICE) ~%s+ C(LOCATION_recode, Treatment(reference='Glenview'))"%'+'.join(xvars_num), 
                  data=subset).fit()
sm_ols_contrast_a.summary()

import re

#coeffs={}
moran={}

## the slope of the regression line here is equivalent to the Moran's I statistic. There appears to be a pattern:
## high prices (i.e. high z) tend to correlate with *lagged* high prices--and lagged high prices mean neighbors per the
## weights matrix.

morandict = {'I':'Morans I statistic',
'vI':'Morans I variance',
'eI':'Morans I expectation',
'zI':'Morans I standardized value',
'p_norm':'P-value'}

def moranplotter():
    mor = ps.Moran(y=np.log(subset.PRICE).values, w=w, transformation = 'r',)
    zx = mor.z
    zy = ps.lag_spatial(w, mor.z)
    fit = ps.spreg.OLS(zy[:, None], zx[:,None])
    ax = sns.regplot(x=zx, y=zy)
    ax.set(xlabel='z-value of I', ylabel='Lagged z-value of I')
    title('Moran\'s I: {moran_i:03.2f}; Moran\'s Expected I: {moran_ei:03.2f}'.format(moran_i=mor.I,moran_ei=mor.EI))
moranplotter()

coeffs={}


y,X = patsy.dmatrices("np.log(PRICE) ~%s+ C(LOCATION_recode, Treatment(reference='Glenview'))"%'+'.join(xvars_num), 
                  data=subset,return_type='dataframe')

## design matrix version
sm_ols_contrast_b = sm.OLS(y, X).fit(cov_type='HC3')

## tokenize location to get just the actual name--extract from unwieldy string
## both lookahead and lookbehind--get content wrapped between T. and ]

ptrn = '(?<=T\.)(.*?)(?=\])' 
m=re.compile(ptrn)
sm_ols_contrast_b_params={}
sm_ols_contrast_b_tvalues={}
for k,v in sm_ols_contrast_b.params.to_dict().iteritems():
    if 'Treatment' in k and 'LOCATION' in k:
        sm_ols_contrast_b_params[m.search(k).group()]=v
    else:
        sm_ols_contrast_b_params[k]=v
for k,v in sm_ols_contrast_b.tvalues.to_dict().iteritems():
    if 'Treatment' in k and 'LOCATION' in k:
        sm_ols_contrast_b_tvalues[m.search(k).group()]=v
    else:
        sm_ols_contrast_b_tvalues[k]=v
        

coeffs[('0: sm_ols_contrast','params')]=pd.Series(sm_ols_contrast_b_params).rename(index={'Intercept':'CONSTANT'})
coeffs[('0: sm_ols_contrast','tstats')]=pd.Series(sm_ols_contrast_b_tvalues).rename(index={'Intercept':'CONSTANT'})

## dummy version instead--notice the R squared in the .29 territory as opposed to .66 for the contrast version

sm_ols_dummy = sm.ols("np.log(PRICE) ~%s"%'+'.join(xvars_num+fulldummylist), 
                  data=subset).fit(cov_type='HC3')

ptrn2 = '(?<=LOCATION\_recode\_)(.*)' 
m=re.compile(ptrn2)
sm_ols_dummy_params={}
sm_ols_dummy_tvalues={}

for k,v in sm_ols_dummy.params.to_dict().iteritems():
    if k.startswith('LOCATION'):
        sm_ols_dummy_params[m.search(k).group()]=v
    else:
        sm_ols_dummy_params[k]=v
for k,v in sm_ols_dummy.tvalues.to_dict().iteritems():
    if k.startswith('LOCATION'):
        sm_ols_dummy_tvalues[m.search(k).group()]=v
    else:
        sm_ols_dummy_tvalues[k]=v

coeffs[('1: sm_ols_dummy','params')]=pd.Series(sm_ols_dummy_params).rename(index={'Intercept':'CONSTANT'})
coeffs[('1: sm_ols_dummy','tstats')]=pd.Series(sm_ols_dummy_tvalues).rename(index={'Intercept':'CONSTANT'})

## pysal plain
ps_ols = ps.spreg.OLS(np.log(subset.PRICE).values[:, None], 
          subset[xvars_num+fulldummylist].values, \
          w=w, 
                  spat_diag=True, \
          name_x=subset[xvars_num+fulldummylist].columns.tolist(), name_y='ln(price)') 
ps_ols_x=[x.replace('LOCATION_recode_','') if x.startswith('LOCATION') else x for x in ps_ols.name_x]

coeffs[('2: ps_ols','params')]=pd.Series(ps_ols.betas[:,0],index=ps_ols_x)
coeffs[('2: ps_ols','tstats')]=pd.Series(pd.DataFrame(ps_ols.t_stat)[0].values,index=ps_ols_x)

## endogenous lag--pysal uses moments to estimate since it violates OLS rules

ps_lag = ps.spreg.GM_Lag(y=np.log(subset.PRICE).values[:, None], 
                         x=subset[xvars_num+fulldummylist].values, \
                  w=w, spat_diag=True, \
                  name_x=subset[xvars_num+fulldummylist].columns.tolist(), name_y='ln(price)') 

ps_lag_x=[x.replace('LOCATION_recode_','') if x.startswith('LOCATION') else x for x in ps_lag.name_x]+['_lag_y']

coeffs[('3: ps_gm_endog','params')]=pd.Series(ps_lag.betas[:,0],index=ps_lag_x)
coeffs[('3: ps_gm_endog','tstats')]=pd.Series(pd.DataFrame(ps_lag.z_stat)[0].values,index=ps_lag_x)
print ps_lag.summary

combomodels = pd.concat(coeffs).reset_index(name='value').rename(columns={'level_0':'model', 'level_1':'variable','level_2':'parameter'}).set_index(['model','variable','parameter']).unstack(level=['variable','model']).value.sort_index(axis=1)
combomodels.params.sort_values('3: ps_gm_endog')

# ps.lag_spatial(w,y)

# X_v = X.assign(w_pool=ps.lag_spatial(w_pool, yxs['pool'].values))



