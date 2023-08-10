#
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import datetime as dtm
import matplotlib.dates as mpd
import pytz
tzutc = pytz.timezone('UTC')

#import operator
import math
import random
import numpy
import scipy
import scipy.optimize as spo
import itertools
import sys
#import scipy.optimize as spo
import os
import operator
#from PIL import Image as ipp
import multiprocessing as mpp
#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#import functools
#
#import shapely.geometry as sgp
#
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from geographiclib.geodesic import Geodesic as ggp
#
#
#import ANSStools as atp
from yodiipy import ANSStools as atp
#import bindex
import contours2kml
import globalETAS as gep
from eq_params import *
#import roc_generic            # we'll eventually want to move to a new library of roc tools.
import random
import geopy
#
#colors_ =  mpl.rcParams['axes.color_cycle']
colors_ = ['b', 'g', 'r', 'c', 'm', 'y', 'k']		# make sure these are correct...
#
emc = {'lat':32.13, 'lon':-115.30, 'event_date':dtm.datetime(2010,4,4,22,40,41, tzinfo=pytz.timezone('UTC'))}
nepal_epi_lon = 84.698
nepal_epi_lat = 28.175

#
# set up dict objects for different earthquakes. maybe from eq_params.py ?
param_keys = ['lat_center', 'lon_center', 'to_dt', 'mc', 'mc_auto', 'cat_len_plus']
nepal_params = {key:val for key,val in zip(param_keys, [nepal_epi_lat, nepal_epi_lon, 
                                                        dtm.datetime(2015,4,30, tzinfo=pytz.timezone('UTC')),
                                                        4.0, 4.5, 220])}
emc_params =   {key:val for key,val in zip(param_keys, [emc['lat'], emc['lon'],
                                                      emc['event_date'] + dtm.timedelta(days=5), 2.5, 5.5, 220])}

class LD_figure(object):
    def __init__(self, lat_center=nepal_epi_lat, lon_center=nepal_epi_lon, d_lat_auto=.5, d_lon_auto=.5,
                       to_dt=dtm.datetime(2015,4,30, tzinfo=pytz.timezone('UTC')), dt_0=30, mc=2.5,
                 mc_auto=4.5, d_lambda=1.76, dm=1.0, Ds=[1.0, 2.0], cat_len_plus=250.):
        '''
        # we want a liner density vs ETAS estimated linear densit figure.
        # first, get a catalog. find the biggest event in the catalog, plot linear distance from that event.
        # then, figure out the ETAS paramaters and see what we get and show for the domain 1. < D < 2.
        '''
        #
        #eqcat = atp.catfromANSS(lon=lons, lat=lats, minMag=cm, dates0=dates, Nmax=None, fout=None, rec_array=True)
        #eqcat.sort(order='event_date')
        # auto_cat(lon_center=None, lat_center=None, d_lat_0=.25, d_lon_0=.5, dt_0=10,  mc=2.5, mc_0=4.5, to_dt=None, catlen=5.0*365.0 range_factor=5., rec_array=True, **kwargs)
        eq_cat_params = atp.auto_cat_params(lat_center=lat_center, lon_center=lon_center,
                                            d_lat=d_lat_auto, d_lon=d_lon_auto, dt_0=dt_0, mc_0=mc_auto, 
                                            to_dt=to_dt, range_factor=5.)
        print('eq_cat_params: ', eq_cat_params)
        # r_val = mpd.num2date(mpd.datestr2num(date_in.astype(str)), tz=tz_out)
        eqcat = atp.catfromANSS(lat=eq_cat_params['lat'], lon=eq_cat_params['lon'], 
                                minMag=mc, dates0=[dtm64_to_datetime(eq_cat_params['mainshock_date']), 
                                                   dtm64_to_datetime(eq_cat_params['mainshock_date']) + dtm.timedelta(days=cat_len_plus)],
                                fout=None, rec_array=True)
        #
        # find the biggest event; dump everything before that.
        #eqcat.sort(order='mag')   # ... except that we want the index too. so let's just spin it:
        mainshock = eqcat[0]
        mainshock_index = 0
        for j,rw in enumerate(eqcat):
            if rw['mag']>mainshock['mag']: mainshock
            #
        #
        eqcat.sort(order='event_date')
        #
        # get a measured dm:
        try:
            m1 = sorted(eqcat['mag'])[-2]
            dm_meas = mainshock['mag']-m1
        except:
            dm_meas=None
        dm_meas = (dm_meas or dm)
        #
        # now, we need a distance formula. this can be using geographiclib or spherical dists. we could hook up some
        # of the class structures in the various etas codes...
        #dists = []
        #for j,ev in enumerate(eqcat):
        #    g1=ggp.WGS84.Inverse(self.mainshock['lat'], self.mainshock['lon'], rw['y'], rw['x'])
        #    r=g1['s12']/1000.)
        #
        dists = [[j,ggp.WGS84.Inverse(mainshock['lat'], mainshock['lon'], ev['lat'], ev['lon'])['s12']/1000.] 
                 for j,ev in enumerate(eqcat)]
        #
        dists2 =[[k,j,r] for k, (j,r) in enumerate(sorted(dists, key=lambda rw: rw[1]))]
        #
        self.__dict__.update(locals())
        #
    def plot_mean_N_prime(self, fignum=0, d_lambda=None, dm=None, dm_meas=None, mc=None):
        #
        f_size = (12,6)
        mainshock=self.mainshock
        d_lambda = (d_lambda or self.d_lambda)
        d_lambda = (d_lambda or 1.76)
        
        dm = (dm or self.dm)
        mc = (mc or self.mc)
        #
        dm_meas = (dm_meas or self.dm_meas)
        #
        fg = plt.figure(figsize=f_size)
        plt.clf()
        ax1 = fg.add_axes([.1,.1,.35,.8])
        ax2 = fg.add_axes([.5,.1,.35,.8])
        #
        ax1.plot(*zip(*[[r,k] for k,j,r in self.dists2]), '.-')
        ax1.set_ylabel('Number of events, $N$')
        ax1.set_xlabel('Distance $r$')
        #
        #plt.figure(figsize=f_size)
        #plt.clf()
        #ax2=plt.gca()
        X,Y = zip(*[[r,k/r] for k,j,r in self.dists2 if r!=0])
        ax2.plot(X,Y, marker='.', ls='-')
        #
        # mark the (approximate) rupture boundary:
        ax2.plot(numpy.ones(2)*.5*10**(.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='--', lw=2., marker='')
        ax2.plot(numpy.ones(2)*1.0*10**(0.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #ax2.plot(numpy.ones(2)*(1.0*10**(.5*mainshock['mag'] - d_lambda) + 1.0*10**(.5*6.3 - d_lambda)), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #
        Y_D2   = 10.**self.log_N_prime_max(m=None, dm=None, D=2.0, mc=mc)
        Y_D1   = 10.**self.log_N_prime_max(m=None, dm=None, D=1.0, mc=mc)
        Y_D2_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=2.0, mc=mc)
        Y_D1_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=1.0, mc=mc)

        ax2.plot([min(X), max(X)], [Y_D2, Y_D2], 'm-' )
        ax2.plot([min(X), max(X)], [Y_D1, Y_D1], 'm--' )
        ax2.plot([min(X), max(X)], [Y_D2_b, Y_D2_b], 'c-' )
        ax2.plot([min(X), max(X)], [Y_D1_b, Y_D1_b], 'c--' )

        #
        #ax2.set_yscale('log')
        #ax2.set_xscale('log')
        #
        #self.__dict__.update(locals())

    def plot_mean_N_prime(self, fignum=0, d_lambda=None, dm=None, dm_meas=None, mc=None):
        #
        f_size = (12,6)
        mainshock=self.mainshock
        d_lambda = (d_lambda or self.d_lambda)
        d_lambda = (d_lambda or 1.76)
        
        dm = (dm or self.dm)
        mc = (mc or self.mc)
        #
        dm_meas = (dm_meas or self.dm_meas)
        #
        fg = plt.figure(figsize=f_size)
        plt.clf()
        ax1 = fg.add_axes([.1,.1,.35,.8])
        ax2 = fg.add_axes([.5,.1,.35,.8])
        #
        ax1.plot(*zip(*[[r,k] for k,j,r in self.dists2]), '.-')
        ax1.set_ylabel('Number of events, $N$')
        ax1.set_xlabel('Distance $r$')
        #
        #plt.figure(figsize=f_size)
        #plt.clf()
        #ax2=plt.gca()
        X,Y = zip(*[[r,k/r] for k,j,r in self.dists2 if r!=0])
        ax2.plot(X,Y, marker='.', ls='-')
        #
        # mark the (approximate) rupture boundary:
        ax2.plot(numpy.ones(2)*.5*10**(.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='--', lw=2., marker='')
        ax2.plot(numpy.ones(2)*1.0*10**(0.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #ax2.plot(numpy.ones(2)*(1.0*10**(.5*mainshock['mag'] - d_lambda) + 1.0*10**(.5*6.3 - d_lambda)), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #
        Y_D2   = 10.**self.log_N_prime_max(m=None, dm=None, D=2.0, mc=mc)
        Y_D1   = 10.**self.log_N_prime_max(m=None, dm=None, D=1.0, mc=mc)
        Y_D2_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=2.0, mc=mc)
        Y_D1_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=1.0, mc=mc)

        ax2.plot([min(X), max(X)], [Y_D2, Y_D2], 'm-' )
        ax2.plot([min(X), max(X)], [Y_D1, Y_D1], 'm--' )
        ax2.plot([min(X), max(X)], [Y_D2_b, Y_D2_b], 'c-' )
        ax2.plot([min(X), max(X)], [Y_D1_b, Y_D1_b], 'c--' )
        
        #ax2.set_yscale('log')
        #ax2.set_xscale('log')

    def plot_N_prime(self, fignum=0, d_lambda=None, dm=None, dm_meas=None, mc=None):
        #
        f_size = (12,6)
        mainshock=self.mainshock
        d_lambda = (d_lambda or self.d_lambda)
        d_lambda = (d_lambda or 1.76)
        
        dm = (dm or self.dm)
        mc = (mc or self.mc)
        #
        dm_meas = (dm_meas or self.dm_meas)
        #
        fg = plt.figure(figsize=f_size)
        plt.clf()
        ax1 = fg.add_axes([.1,.1,.35,.8])
        ax2 = fg.add_axes([.5,.1,.35,.8])
        #
        N = len(self.dists2)
        ax1.plot(*zip(*[[r,k] for k,j,r in self.dists2]), '.-')
        #ax1.plot(*zip(*[[r,N-k] for k,j,r in self.dists2]), '.-')
        ax1.set_ylabel('Number of events, $N$')
        ax1.set_xlabel('Distance $r$')
        #
        #plt.figure(figsize=f_size)
        #plt.clf()
        #ax2=plt.gca()
        X,Y = zip(*[[r,1./(r-self.dists2[j][2])] for k,j,r in self.dists2[1:] if (r-self.dists2[j][2]>0)])
        ax2.plot(X,Y, marker='.', ls='-')
        #
        # mark the (approximate) rupture boundary:
        ax2.plot(numpy.ones(2)*.5*10**(.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='--', lw=2., marker='')
        ax2.plot(numpy.ones(2)*1.0*10**(0.5*mainshock['mag'] - d_lambda), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #ax2.plot(numpy.ones(2)*(1.0*10**(.5*mainshock['mag'] - d_lambda) + 1.0*10**(.5*6.3 - d_lambda)), [min(Y), max(Y)], ls='-', lw=2., marker='')
        #
        Y_D2   = 10.**self.log_N_prime_max(m=None, dm=None, D=2.0, mc=mc)
        Y_D1   = 10.**self.log_N_prime_max(m=None, dm=None, D=1.0, mc=mc)
        Y_D2_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=2.0, mc=mc)
        Y_D1_b = 10.**self.log_N_prime_max(m=None, dm=dm_meas, D=1.0, mc=mc)

        ax2.plot([min(X), max(X)], [Y_D2, Y_D2], 'm-' )
        ax2.plot([min(X), max(X)], [Y_D1, Y_D1], 'm--' )
        ax2.plot([min(X), max(X)], [Y_D2_b, Y_D2_b], 'c-' )
        ax2.plot([min(X), max(X)], [Y_D1_b, Y_D1_b], 'c--' )

        #
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        #
        #self.__dict__.update(locals())
        
    def GR_dist(self, fignum=None):
        N = len(self.eqcat)
        GR = [[m, N-j] for j,m in enumerate(sorted(self.eqcat['mag']))]
        #
        if fignum is not None:
            plt.figure(fignum)
            plt.clf()
            ax=plt.gca()
            ax.plot(*zip(*GR), ls='-', marker='.')
            #
            ax.set_yscale('log')
    def log_N_prime_max(self, m=None, mc=None, dm=1.0, D=1.5, d_lambda=1.76):
        m = (m or self.mainshock['mag'])
        mc = (mc or self.mc)
        d_lambda = (d_lambda or self.d_lambda)
        dm = (dm or self.dm)
        #print('***', m, mc, dm)
        #
        return numpy.log10(2) - (.5*m - d_lambda) + (2./(2.+D))*numpy.log10(1.+D/2.) + (D/(2.+D))*(m - dm - mc)

# eventually, consolidate the generalized datetime-wrangler code.
def dtm64_to_datetime(x, tz_out=pytz.timezone('UTC')):
    if isinstance(x, dtm.datetime): return x
    return  mpd.num2date(mpd.datestr2num(x.astype(str)), tz=tz_out)

def log_N_prime_max(m, mc, dm=1.0, D=1.5, d_lambda=1.76):
    return numpy.log10(2) - (.5*m - d_lambda) + (2./(2.+D))*numpy.log10(1.+D/2.) + (D/(2.+D))*(m - dm - mc)

A = LD_figure(**nepal_params)
A.plot_mean_N_prime()

A.plot_N_prime()
# lats=[28.175-1., 28.175+1.], lons = [84.7-1., 84.7+1], to_dt=dtm.datetime(2015,5,1, tzinfo=pytz.timezone('UTC')) 
#
A = LD_figure(**nepal_params)
A.plot_mean_N_prime()

A = LD_figure(**emc_params)

A.plot_mean_N_prime()
A.plot_N_prime()

A.GR_dist(fignum=1)



