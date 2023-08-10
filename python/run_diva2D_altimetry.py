import os
import glob
import logging
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.basemap import Basemap
import pydiva2d
import netCDF4
import divaaltimetry
import matplotlib.pyplot as plt

logger = logging.getLogger('divaAltimetry')
logger.setLevel(logging.DEBUG)

divadir = "/home/ctroupin/Software/DIVA/DIVA-diva-4.7.1"
projectdir = "/home/ctroupin/Projects/Altimetry-Interpolation/"
coastfile = os.path.join(projectdir, "diva/coast.cont")
datadir = os.path.join(projectdir, "data/")
paramfile = os.path.join(projectdir, "diva/param.par")
paramfilemesh = os.path.join(projectdir, "diva/param.par.mesh")
outputdir = os.path.join(projectdir, "results/diva2D/")
figdir = "../figures/diva2d/"
if not os.path.exists(divadir):
    logger.error("Diva directory doesn't exist")
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    logger.debug("Create output directory")
if not os.path.exists(figdir):
    os.makedirs(figdir)
    logger.debug("Create figure directory")

rcParams.update({'font.size': 14, 'figure.dpi': 300, 'savefig.bbox': 'tight'})
coordinates = (-6.75, 36.001, 30, 48.)
meridians = np.arange(-8., 40., 8.)
parallels = np.arange(30., 50., 4.5)

m = Basemap(projection='merc',
            llcrnrlon=-6., llcrnrlat=30.,
            urcrnrlon=40., urcrnrlat=48.,
            lat_ts=39., resolution='i')

DivaDirs = pydiva2d.DivaDirectories(divadir)
DivaFiles = pydiva2d.Diva2Dfiles(DivaDirs.diva2d)

mesh2d = pydiva2d.Diva2DMesh().make(divadir, contourfile=coastfile, paramfile=paramfilemesh)

mesh2d.describe()

for datafile in sorted(glob.glob(os.path.join(datadir, 'data_20140901*.dat'))):
    
    
    outputfile = "".join((os.path.basename(datafile).split('.')[0], '.nc'))
    figname = os.path.basename(datafile).split('.')[0]
    logger.info("Output file: {0}".format(outputfile))
    
    """
    results2d = pydiva2d.Diva2DResults().make(divadir, datafile=datafile,
                                              paramfile=paramfile, 
                                              contourfile=coastfile,
                                              outputfile=os.path.join(outputdir, outputfile))
    """
    
    # Make plot
    SLA = divaaltimetry.AltimetryField().from_diva2d_file(os.path.join(outputdir, outputfile))
    SLA.add_to_plot(figname=os.path.join(figdir, figname), m=m,
                    meridians=meridians, parallels=parallels,
                    vmin=-0.2, vmax=0.2,
                    cmap=plt.cm.RdYlBu_r)
    #divadata = pydiva2d.Diva2DData().read_from(datafile)

