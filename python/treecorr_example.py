get_ipython().magic('pylab inline')
import numpy as np
import fastcat as fc
import treecorr as tc
import healpy as hp
import os
try:
    path=os.environ['FASTCAT_MOCKS']
except KeyError:
    path="./"

data1=fc.Catalog(read_from=path+"160329/simple/catalog0.h5")
data2=fc.Catalog(read_from=path+"160329/noDither/catalog0.h5")

# Let's see what are types of our window func and photoz objects
print "Window obj strings:",data1.window.typestr, data2.window.typestr
print "Window obj types: ", type(data1.window),type(data2.window)
print "PhotoZ obj strings:",data1.photoz.typestr,data2.photoz.typestr
print "PhotoZ obj types: ",type(data1.photoz), type(data2.photoz)

print "HEALPIX info string:",data2.window.info

# Let's look at some of the photo-z interfaces
# PhotoZ object
pz=data1.photoz
# first let's take just first five galaxies
arr5=data1[[np.random.uniform(len(data1[:])) for i in range(5)]]

# Mean and RMS
pz.getMeanRMS(arr5)

# get mins and maxes
pz.getMinMax(arr5)

# get probabilities at z=1.0 +- (0.01)
pz.PofZ(arr5,1.0,0.01)

## courtsy of fhaviersanchez
def make_hp_map(nside,data):
    import healpy as hp
    pix_nums = hp.ang2pix(nside,np.pi/2-data['dec']*np.pi/180,data['ra']*np.pi/180)
    bin_count = np.bincount(pix_nums)
    map_gal = np.append(bin_count,np.zeros(12*nside**2-len(bin_count)))
    return map_gal

def cutDataMakeRandom(data):
    # cut data to some small redshift bin
    zcent=0.4
    zbinsize=0.1    
    cutdata=data[np.where(abs(data['z']-zcent)<zbinsize/2)]
    print "Picked",len(cutdata),"from",len(data.data),"objects."
    ran=np.hstack((cutdata,)*10)
    Nr=len(ran)
    ran['ra']=np.random.uniform(0,360,Nr)
    ran['dec']=np.degrees(np.arcsin(np.random.uniform(-1,1,Nr)))
    ran=data.window.applyWindow(ran)
    figure(figsize=(12,12))
    #subplot(2,1,1)
    hp.mollview(make_hp_map(32,cutdata),sub=(1,2,1))
    #subplot(2,1,2)
    hp.mollview(make_hp_map(32,ran),sub=(1,2,2))
    return cutdata,ran

cdata1,ran1=cutDataMakeRandom(data1)

cdata2,ran2=cutDataMakeRandom(data2)

def getCorrelation(cutdata,ran):
    #treecorr catalogs
    cat=tc.Catalog(ra=cutdata['ra'],dec=cutdata['dec'],ra_units='degrees',dec_units='degrees')
    rcat=tc.Catalog(ra=ran['ra'],dec=ran['dec'],ra_units='degrees',dec_units='degrees')
    ## correlators
    dd=tc.NNCorrelation(min_sep=0.001,bin_size=0.1,max_sep=40., sep_units='degrees')
    dd.process(cat)
    dr=tc.NNCorrelation(min_sep=0.001,bin_size=0.1,max_sep=40., sep_units='degrees')
    dr.process(cat,rcat)
    rr=tc.NNCorrelation(min_sep=0.001,bin_size=0.1,max_sep=40., sep_units='degrees')
    rr.process(rcat,rcat)
    xi,xivar=dd.calculateXi(rr,dr)
    logr,meanlogr=dd.logr, dd.meanlogr
    rr=exp(logr)
    return rr,xi,sqrt(xivar)

r1,xi1,xie1=getCorrelation(cdata1,ran1)
r2,xi2,xie2=getCorrelation(cdata2,ran2)

# plot measured correaltion functions
# we seem to have a weird issue with normalization
pylab.figure(figsize=(13,13))
pylab.plot(r1,xi1*r1,'ro-')
pylab.plot(r2,xi2*r2*1.29,'bo-')
pylab.xlabel('$\\theta$')
pylab.ylabel('$\\xi(\\theta)$')
#pylab.ylim(1e-9,2e-5)
pylab.semilogx()



