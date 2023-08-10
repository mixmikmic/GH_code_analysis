import numpy as np
import numpy.ma as ma

import iris

def latitude_line(lat0, lat):
    """ 
    Define the indices which mark a latitude. 
    @author: Aleksi Nummelin
    """
    iind=[]
    jind=[]
    sum=0
    i=0 #keeps track of the model i index
    i2=0 #keeps track of the length of the jind and iind
    i3=0 #helper index for looping backwards
    maxy=lat.shape[0]
    maxx=lat.shape[1]-1
    keep_looping=True
    backwards=False
    bipolar=False
    if len(np.where(np.diff(lat[-1,:])==0)[0])==1:
        bipolar=True
    while keep_looping:
        if not backwards and (lat0<max(lat[:,i]) and lat0>=min(lat[:,i])):
            #if the latitude is available append the index, this is the normal situation
            ind=np.where(lat0<=lat[:,i])[0][0] #(np.logical_and((lat-l)>=(-.5*dlat), (lat-l)<(.5*dlat)))
            jind.append(ind)
            iind.append(i)
            i=i+1; i2=i2+1; i3=i3+1
        elif len(jind)>0 and bipolar: #not (lat0<ma.max(lat[:,i:]) and lat0>=ma.min(lat[:,i:])):
            #if the latitude doesn't exist and some indices are already there (situation close to north pole in in bipolar grid)
            #Also check that the latitude doesn't exist in the rest of the matrix (which can be the case for the tripolar setup)
            #Then loop backwards
            if (lat0<max(lat[:,i-1]) and lat0>=min(lat[:,i-1])):
                #ind=np.round(np.interp(lat0, lat[jind[i3-1]:,i-1], np.arange(jind[i3-1],maxy)))
                ind=np.where(lat0<=lat[:,i-1])[0][-1]
                jind.append(ind)
                iind.append(i-1)
                i2=i2+1; i3=i3-1
            else:
                keep_looping=False
                #fill in the the list if needed
                if jind[-1]-jind[0]>1:
                    kk=jind[-1]-jind[0]
                for k in range(kk):
                    jind.append(jind[-1]-1)
                    iind.append(iind[-1])
            i=i-1;
            backwards=True
        else:
            i=i+1;
        if i>maxx or i<0:
            keep_looping=False
    #
    return iind, jind

def calc_heat_trasport1(iind,jind,xtransport,ytransport):
    """ 
    calculate the heat transport accross a given line.
    calculate first iind and jiind. Note that this will work
    in a cartesian grid and on a NorESM type of C grid.
    #
    author: Aleksi Nummelin    
    """
    #looks already pretty good some things should be still figured out
    #First cell
    sumtot=ytransport[:,jind[0],iind[0]]
    if jind[1]>jind[0]:
        #if the next step is up right then add the transport from the cell to the right
        sumtot=ma.sum([sumtot,-1*xtransport[:,jj,ii+1]],0)
    #Last cell
    if iind[-1]==xtransport.shape[-1]-1:
        #if normal case with increasing indices
        if jind[-1]==jind[0]:
            sumtot=ma.sum([sumtot, ytransport[:,jind[-1],iind[-1]]],0)
        elif jind[-1]>jind[0]:
            sumtot=ma.sum([sumtot, ytransport[:,jind[-1],iind[-1]]+xtransport[:,jind[0],iind[0]]],0)
        elif jind[-1]<jind[0]:
            sumtot=ma.sum([sumtot, ytransport[:,jind[-1],iind[-1]]-xtransport[:,jind[0],iind[0]]],0)
    #if a tripolar grid
    elif iind[-1]>iind[-2] and jind[-1]>jind[-2]:
        sumtot=ma.sum([sumtot, ytransport[:,jind[-1],iind[-1]]-xtransport[:,jind[-1],iind[-1]]],0)
    ##########################
    # - LOOP OVER THE REST - #
    ##########################
    for j in range(1,len(jind)-1):
        #note that the last point is the copy of the first in case of bibolar
        jj=jind[j]; ii=iind[j]
        ##################################
        #Straight Line in X
        if jind[j-1]==jj and iind[j-1]<ii:
            #add the transport from the cell below
            sumtot=ma.sum([sumtot, ytransport[:,jj,ii]],0)
            if jind[j+1]>jj:
                #if the cell is last one in a strike of a cells before a step upwardright
                sumtot=ma.sum([sumtot, -1*xtransport[:,jj,ii+1]],0)
        ###################################
        #Straight backward line in x
        elif jind[j-1]==jj and iind[j-1]>ii and jj+1<ytransport.shape[1]:
            #add the transport from the cell above
            sumtot=ma.sum([sumtot, -1*ytransport[:,jj+1,ii]],0)
            if jind[j+1]<jj and iind[j+1]<ii:
                #if the cell is last one in a strike of a cells before a step downleft add the positive of xtransport
                sumtot=ma.sum([sumtot, xtransport[:,jj,ii-1]],0)
        ###################################
        #Straight line in y downwards
        if jind[j-1]>jj and iind[j-1]==ii:
            sumtot=ma.sum([sumtot, xtransport[:,jj,ii]],0)
            if iind[j+1]>ii:
                #if the cell is last one in a strike of a cells before a step right add the ytransport from below
                sumtot=ma.sum([sumtot, ytransport[:,jj,ii]],0)
        ###################################
        #Straight line in y upwards
        if jind[j-1]<jj and iind[j-1]==ii:
           sumtot=ma.sum([sumtot, -1*xtransport[:,jj,ii+1]],0)
           if iind[j+1]<ii and jj+1<xtransport.shape[-2]:
               #if the cell is last one in a strike of a cells before a step left add the ytransport from above
               sumtot=ma.sum([sumtot, -1*ytransport[:,jj+1,ii]],0)
        ###################################
        #Step down-right
        elif jind[j-1]>jj and iind[j-1]<ii:
            #add transport from the cell to the left
            sumtot=ma.sum([sumtot,xtransport[:,jj,ii]],0)
            if iind[j+1]!=ii:
                #and if the next move is away from this point ie the next cell is not the cell below
                #then add also the transport from below
                sumtot=ma.sum([sumtot,ytransport[:,jj,ii]],0)
        ####################################
        #Step upright
        elif jind[j-1]<jj and iind[j-1]<ii:
            #Add the ytransport from cell below
            sumtot=ma.sum([sumtot,ytransport[:,jj,ii]],0)
            if jind[j+1]!=jj:
                #and if the next step is not next to it then negative of the x transport from the cell to the right
                sumtot=ma.sum([sumtot,-1*xtransport[:,jj,ii+1]],0)
                if iind[j+1]<ii:
                #if the next step is step up-left (ie you're in the turning point to backward stepping)
                    sumtot=ma.sum([sumtot,-1*ytransport[:,jj+1,ii]],0)
        #####################################
        #Step up-left (backwards up)
        elif jind[j-1]<jj and iind[j-1]>ii:
            #add x transport from the cell to the right
            sumtot=ma.sum([sumtot,-1*xtransport[:,jj,ii+1]],0)
            if iind[j+1]<ii and jj+1<ytransport.shape[1]:
            #if the next step is not directly above add the transport from the cell above
                sumtot=ma.sum([sumtot,-1*ytransport[:,jj+1,ii]],0)
                if jind[j+1]<jj:
                #and if the next step is down left then add transport from the cell to the left
                    sumtot=ma.sum([sumtot,xtransport[:,jj,ii]],0)
        ######################################
        #Step down-left (backwards down)
        elif jind[j-1]>jj and iind[j-1]>ii:
            #add y transport from above
            sumtot=ma.sum([sumtot,-1*ytransport[:,jj+1,ii]],0)
            if jind[j+1]<jj:
                #and if the next cell is not the cell to the left add x transport from the cell to the left
                sumtot=ma.sum([sumtot,xtransport[:,jj,ii]],0)
    #
    return sumtot

def calc_heat_transport2(lon,lat,hfx,hfy):
    """
    Code is a snippet of the working code to calculate heat transport in NorESM.
    It requires hfy and hfx and the latitudes (hfy, hfx are timeseries of 2D fields). 
    #
    Created on Fri Sep  9 09:26:08 2016
    #
    @author: Stephen Outten
    """
    #
    dlat = 1 #this can be anything, but should probably be model resolution or coarser
    lati = np.arange(-90,90+dlat,dlat)
    htro = np.zeros((hfy.shape[0], len(lati)))
    iinds=[]; jinds=[]
    countind = []
    for j,lat0 in enumerate(lati):
        iind,jind = latitude_line(lat0, plat)
        iinds.append(iind)
        jinds.append(jind)
        countind.append(len(iind))
        if len(iind)>0:
        # hfx comes from next cell thus 2 hfxs needed, one shifted by a cell for -1 values
            iind = np.array(iind);  jind = np.array(jind)    # Arrays are so much more useful
            jdiff = np.ones(len(jind)) * np.nan      # ***** HTRO with compelte line
            jdiff[0:-1] = jind[1:] - jind[0:-1]     #  ***** All these lines
            jdiff[-1] = jind[0] - jind[-1]
            hfx_line = hfx[:,jind,iind]
            hfx_shift = np.zeros(hfx_line.shape)
            hfx_shift[:,0:-1] = hfx[:,jind[0:-1],iind[0:-1]+1]  # create a shifted line with same jind but +1 iind
            hfx_shift[:,-1] = hfx[:,jind[-1],iind[0]]        # last value is jind of last box but iind of first box
            hfxflag1 = np.zeros(len(jdiff))
            hfxflag2 = np.zeros(len(jdiff))
            hfxflag1_array = np.where(jdiff<0)[0]+1   # account for last element being different and change the first element instead
            hfxflag1_array[np.where(hfxflag1_array==len(hfxflag1))] = 0
            hfxflag1[hfxflag1_array] = 1
            hfxflag2[np.where(jdiff>0)[0]] = -1
            hfyflag = np.ones(len(jind))
            #comment by Aleksi: I think you might want to modify the line below so that all the additons are done with nansum
            #this is because np.nan+number gives np.nan which is not desired
            total_lat = np.nansum(hfyflag*hfy[:,jind,iind] + hfxflag1*hfx_line + hfxflag2*hfx_shift,1)
            #so this might be more correct
            #total_lat = np.nansum(np.nansum([hfyflag*hfy[:,jind,iind], hfxflag1*hfx_line, hfxflag2*hfx_shift],0),1)
            htro[:,j] = total_lat      
        else:
            htro[:,j] = 0
    return htro

hfx_file = '/g/data/ua6/DRSv2_legacy/CMIP5/NorESM1-M/historical/mon/ocean/r1i1p1/hfx/latest/hfx_Omon_NorESM1-M_historical_r1i1p1_185001-200512.nc'
hfy_file = '/g/data/ua6/DRSv2_legacy/CMIP5/NorESM1-M/historical/mon/ocean/r1i1p1/hfy/latest/hfy_Omon_NorESM1-M_historical_r1i1p1_185001-200512.nc'

hfx_cube = iris.load_cube(hfx_file, 'ocean_heat_x_transport')
hfy_cube = iris.load_cube(hfy_file, 'ocean_heat_y_transport')

print(hfx_cube)

hfy_cube.coord('longitude').points

hfx_cube.coord('longitude').points



