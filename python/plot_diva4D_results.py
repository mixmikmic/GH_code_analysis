import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sys
sys.path.insert(0, "../")
import pydiva4d

pydiva4d.logger.setLevel(logging.DEBUG)

figdir = './figures/BlackSea/'
if not os.path.exists(figdir):
    os.makedirs(figdir)

divamaindir = '/home/ctroupin/Software/DIVA/DIVA-diva-4.7.1/'
Diva4Ddirs = pydiva4d.Diva4DDirectories(divamaindir)

Diva4Dfiles = pydiva4d.Diva4Dfiles(Diva4Ddirs.diva4d)

contourdepth = pydiva4d.Contourdepth()
contourdepth.read_from(Diva4Dfiles.contourdepth)
contourdepth.depthlist

for idepth, depth in enumerate(contourdepth.depthlist):
    
    contourfile = os.path.join(Diva4Ddirs.diva4dparam, 'coast.cont.{0}'.format(str(10001 + idepth)))

    contour2D = pydiva4d.Diva2DContours()
    contour2D.read_from(contourfile)

    fig = plt.figure()
    contour2D.add_to_plot(color='k', linewidth=.5)
    plt.xlim(26., 42.)
    plt.ylim(40., 48.)
    plt.title("Contour at depth: {0} m".format(depth))
    plt.savefig(os.path.join(figdir, "BlackSea_contour{0}".format(idepth)))
    plt.close()

for idepth, depth in enumerate(contourdepth.depthlist):
    # Create the file names
    meshtopofile = os.path.join(Diva4Ddirs.diva4dmesh, "mesh.dat.{0}".format(str(10000 + idepth + 1)))
    meshfile = os.path.join(Diva4Ddirs.diva4dmesh, "meshtopo.{0}".format(str(10000 + idepth + 1)))
    
    # Mesh object
    Mesh = pydiva4d.Diva2DMesh()
    Mesh.read_from(meshfile, meshtopofile)
    
    # Make the plot
    fig = plt.figure()
    ax = plt.subplot(111)
    Mesh.add_to_plot(linewidth=0.1, color='k')
    plt.xlim(26., 42.)
    plt.ylim(40., 48.)
    plt.title("Mesh at depth: {0} m".format(depth))
    plt.savefig(os.path.join(figdir, "BlackSea_mesh{0}".format(idepth)))
    plt.close()

Monthlist = pydiva4d.Monthlist()
Monthlist.read_from(Diva4Dfiles.monthlist)
Monthlist.monthlist

Yearlist = pydiva4d.Yearlist()
Yearlist.read_from(Diva4Dfiles.yearlist)
Yearlist.yearlist

Varlist = pydiva4d.Varlist()
Varlist.read_from(Diva4Dfiles.varlist)
Varlist.varlist

for variables in Varlist.varlist:
    for yearperiods in Yearlist.yearlist:
        for monthperiods in Monthlist.monthlist:
            for idepth, depthlevels in enumerate(contourdepth.depthlist):
                
                # Create file name
                resultfile = "{0}.{1}.{2}.{3}.anl.nc".format(variables, yearperiods,
                                                             monthperiods, str(10000 + idepth + 1))
                
                # Create figure name
                figname = ''.join((resultfile.replace('.', '_'), '.png'))
                figtitle = "{0}, Year: {1}, Months: {2}, Depth: {3} m".format(variables, yearperiods,
                                                                                       monthperiods, str(depthlevels))
                
                # Load the results
                if os.path.exists(os.path.join(Diva4Ddirs.diva4dfields, resultfile)):
                    Results = pydiva4d.Diva2DResults(os.path.join(Diva4Ddirs.diva4dfields, resultfile))
                    fig = plt.figure()
                    ax = plt.subplot(111)
                    resultplot = Results.add_to_plot()
                    plt.colorbar(resultplot)
                    plt.title(figtitle)
                    plt.savefig(os.path.join(figdir, figname))
                    plt.close()



