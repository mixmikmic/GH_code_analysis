import numpy as np
import apogee.tools.read as apread
from matplotlib import pyplot as plt
import pandas as pd
import csv
from apogee.tools import bitmask
import math

def calcR(x,pos1,pos2):
    ccfCenter = max(x)
    primary = np.where(x == ccfCenter)
    peak_loc = primary[0][0]
    #print(type(peak_loc))
    #pos1+= 1
    Mirror = (x[peak_loc:pos2])[::-1]
    #Mirror = (x[ccfCenter:pos2])[::-1]
    sigmaA = np.sqrt(1.0 / (2.0 * len(Mirror)) * np.sum((x[pos1:peak_loc] - Mirror)**2))
    r = np.max(x) / (np.sqrt(2.0) * sigmaA)
    print(r)
    return r

def bisector(xccf,yccf):
    height = max(yccf) - min(yccf)
    slices = height/4.0
    bounds = np.arange(min(yccf),height,slices)
    if len(bounds) != 0:
        z1 = (bounds[0] + bounds[1])/2.0
        z2 = (bounds[1] + bounds[2])/2.0
        z3 = (bounds[2] + bounds[3])/2.0
        z4 = (bounds[3] + bounds[4])/2.0
        y_bisector = np.array([z1,z2,z3,z4])

        x_bisector = []
        x0 = []
        x1 = []
        x2 = []
        x3 = []
        for i in range(len(yccf)):
            if yccf[i] <= bounds[4] and yccf[i] > bounds[3]:
                x0.append(xccf[i])
        x_0 = (np.mean(x0))
        x_bisector.append(x_0)

        i = 0
        for i in range(len(yccf)):
            if yccf[i] <= bounds[3] and yccf[i] >= bounds[2]:
                x1.append(xccf[i])
        x_1=(np.mean(x1))
        x_bisector.append(x_1)

        i = 0
        for i in range(len(yccf)):
            if yccf[i] <= bounds[2] and yccf[i] >= bounds[1]:
                x2.append(xccf[i])
        x_2=(np.mean(x2))
        x_bisector.append(x_2)

        i = 0
        for i in range(len(yccf)):
            if yccf[i] <= bounds[1] and yccf[i] >= bounds[0]:
                x3.append(xccf[i])
        x_3=(np.mean(x3))
        x_bisector.append(x_3)

        bisector_pts = np.vstack([x_bisector,y_bisector])
        #print(bisector_pts)
        return(bisector_pts)
    #else:
        #x_bisector = 0.0
        #y_bisector = 0.0
        #error = np.vstack([x_bisector,y_bisector])
        return(error)

def xrange(x_bisector):
    #print(x_bisector)
    xr = max(x_bisector) - min(x_bisector)
    return xr

#Calculate the R-ratios for the likely_binary function
def r_ratio(r51,r151,r101):
        r1_ratio = r151/r101
        r2_ratio = r101/r51
        R1_ratio = math.log10(r1_ratio)
        R2_ratio = math.log10(r2_ratio)
        ratios = [round(R1_ratio,3),round(R2_ratio,3)]
        return ratios

#def idSB2s(CCF,xr,r51,r151,r101): # cuts to identify SB2s
    #min_r51 = min(r51)
    #min_r101 = min(r101)
    #min_r151 = min(r151)
    #peak_401 = max(CCF) - min(CCF)
    #max_xr = max(xr)
def idSB2s(R1_ratio, R2_ratio,r51,r151,r101,xr): # cuts to identify SB2s from Kevin's IDL Routine
    min_r51 = r51
    min_r101 = r101
    min_r151 = r151
    r1_ratio = R1_ratio
    r2_ratio = R2_ratio
    max_xr = xr
    
    likely_sb2s = np.where((math.log10(r1_ratio) > 0.06 and (math.log10(r1_ratio) < 0.13 and 
                            math.log10(min_r101) < 0.83)) or (math.log10(r2_ratio) > 0.05 and 
                            math.log10(r2_ratio) < 0.02 and math.log10(min_r51) < 0.83) and
                            math.log10(min_r51) > 0.25 and math.log10(min_r101) > 0.22 and
                            math.log10(peak_401) > -0.5 and math.log10(max_xr) < 2.3 and 
                            math.log10(max_xr) > 0.7
                          )
    return likely_sb2s

from astropy.io import fits

allStarDR14 = apread.allStar(rmcommissioning=False,main=False,ak=True,akvers='targ',adddist=False)

locationIDs = allStarDR14['LOCATION_ID']
apogeeIDs = allStarDR14['APOGEE_ID']
apogeeIDs = [s.decode('utf-8') for s in apogeeIDs]

import os.path
from pathlib import Path

for i in range(100):
    locationID = locationIDs[i]
    apogeeID = apogeeIDs[i]
    my_file = Path('/Volumes/coveydata-5/APOGEE_Spectra/APOGEE2_DR14/dr14/apogee/spectro/redux/r8/stars/apo25m/'+str(locationID)+'/'+'apStar-r8-'+str(apogeeID)+'.fits')
    #if my_file.is_file():
        #path = '/Volumes/coveydata-5/APOGEE_Spectra/APOGEE2_DR14/dr14/apogee/spectro/redux/r8/stars/apo25m/'+str(locationID)+'/'+'apStar-r8-'+str(apogeeID)+'.fits'
    try:
        asb_path = my_file.resolve()
    except FileNotFoundError:
        path = '/Volumes/coveydata-5/APOGEE_Spectra/APOGEE2_DR14/dr14/apogee/spectro/redux/r8/stars/apo25m/'+str(locationID)+'/'+'apStarC-r8-'+str(apogeeID)+'.fits'
    else:
        path = '/Volumes/coveydata-5/APOGEE_Spectra/APOGEE2_DR14/dr14/apogee/spectro/redux/r8/stars/apo25m/'+str(locationID)+'/'+'apStar-r8-'+str(apogeeID)+'.fits'

    data = fits.open(path)
    point = data[9]
    xccf = point.data[0][32]
    CCF = point.data[0][27]
    HDU0 = fits.getheader(path,0)
    nvisits = HDU0['NVISITS']
    for visit in range(0,nvisits):
        if nvisits != 1:
            ccf = CCF[visit+2]
            nonzeroes = np.count_nonzero(ccf) # This condition is meant to eliminate visits that are empty
            if nonzeroes >= 1:
                bs_pt = bisector(xccf, ccf)
                x_range = xrange(bs_pt[0])
                print('--Visit '+str(visit)+'--')
                R401 = calcR(ccf,0,400)
                R151 = calcR(ccf,125,275)
                R101 = calcR(ccf,150,250)
                R51 = calcR(ccf,175,225)
                Ratios = r_ratio(R51,R151,R101)
                r1 = Ratios[0]
                r2 = Ratios[1]
                #plt.plot(ccf,label='Visit '+str(visit))
                #plt.legend(loc='upper right')
                #plt.show()
        

