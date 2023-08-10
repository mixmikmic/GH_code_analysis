get_ipython().magic('matplotlib notebook')

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from Storms.parameters import *

radcols=['64ne', '64se', '64sw', '64nw', '50ne', '50se', '50sw', '50nw',
       '34ne', '34se', '34sw', '34nw']

path='../tmp/'

filename='step1.txt'

fk=0.92  # coefficient for going from 1m to 10m in velocities 

inpdat = pd.read_csv(path+filename, delimiter='\t')

inpdat.head()

if inpdat.lon.apply(np.sign).diff().sum() > 0:
    m=inpdat.lon != inpdat.lon[0]
    inpdat.lon[m]+360. if inpdat.lon[0] > 0 else npdat.lon[m]-360.

x=inpdat.lon
y=inpdat.lat

dt=np.gradient(inpdat.time)*3600 # compute dt (translate time from hours to sec)

dx_dt = np.gradient(x,dt)
dy_dt = np.gradient(y,dt)
velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

#velocity

vtrx = velocity[:,0] * deg2m * np.cos(np.radians(inpdat.lat.values))  #adjust for latitude
vtry = velocity[:,1] * deg2m

vtr = np.sqrt(vtrx**2+vtry**2)

#print vtrx,vtry,vtr

ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

tangent = np.array([1/ds_dt] * 2).transpose() * velocity

phi=np.arctan2(tangent[:,1],tangent[:,0]) # the angle of the velocity vector

cosfi = np.cos(phi)
sinfi = np.sin(phi)

# extend dataset to save new data
inpdat['vtrx']=vtrx
inpdat['vtry']=vtry
inpdat['vtr']=vtr
inpdat['cosfi']=cosfi
inpdat['sinfi']=sinfi

inpdat.head()

cols=['w'+ x for x in radcols]

#temp = np.zeros((time.size, 12))
#d = pd.DataFrame(temp, columns = cols)

inpdat

an=np.array([tetaNE, tetaSE, tetaSW, tetaNW,tetaNE, tetaSE, tetaSW, tetaNW,tetaNE, tetaSE, tetaSW, tetaNW])# to be used
sinan = np.sin(np.radians(an+90))  # an +90 = angle of tangential wind
cosan=np.cos(np.radians(an+90))

V0=np.array([64, 64, 64, 64, 50, 50, 50, 50, 34, 34, 34, 34])*kt2ms*fk #translate knots to m/s and from 1km to 10km

R=inpdat.ix[:,radcols].copy()

R=R[R>0]

RATIO = (rmax0/R)**b0    # assume exponential decay eqs (13) from JRC report
EXPRATIO = np.exp(-RATIO)  #                       "

RATIO

VT=vtr[:,np.newaxis]*(cosfi[:,np.newaxis] * cosan + sinfi[:,np.newaxis] * sinan)*(1-EXPRATIO)   # Eq (15) from JRC report

VT

VT.loc[inpdat.lat<0] = -VT # reverse for south hemishpere

VT

VV = V0-VT   # substract translational velocity from TC velocity

deltalatWR=R/deg2m*np.sin(np.radians(an))

deltalatWR

latWR=inpdat.lat[:,np.newaxis]+deltalatWR

latWR

fWR=2*omega*np.abs(np.sin(np.radians(latWR))) # Coriolis parameter f=2*Omega*sin(lat)
Vnco=((VV+R*fWR/2)**2-(R*fWR/2)**2)**0.5

Vnco=Vnco.replace(np.nan,0)

Vnco

#change header
Vnco.columns = cols

# extend dataset to save the velocities
inpdat = pd.concat([inpdat, Vnco], axis=1)

inpdat.head()

vs = inpdat.vmax*fk-vtr

vmax0vt = np.maximum(vs,Vnco.max(axis=1))

inpdat['vmax0vt'] = vmax0vt

inpdat = inpdat.set_index('time')

inpdat.head()

inpdat.to_csv(path+'step2.txt',index=True, sep='\t')

