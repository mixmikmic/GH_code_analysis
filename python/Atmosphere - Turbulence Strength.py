import numpy
from astropy.io import fits
import os
get_ipython().magic('matplotlib notebook')
from matplotlib import pyplot
import bokeh.plotting as bplot
bplot.output_notebook()

from aotools.circle import zernike
from soapy import atmosphere, confParse

SOAPY_CONF = "conf/test_conf.py"

config = confParse.Configurator(SOAPY_CONF)
config.loadSimParams()

config.atmos.wholeScrnSize = 4096
config.atmos.scrnNo = 1

NZERNS = 30
NR0s = 10
NIters = 5000
RUNS = 5
R0s = numpy.linspace(0.05, 0.5, NR0s)

print("Telescope Diameters:{}".format(config.tel.telDiam))

# Make the zernike to use in advance
Zs = zernike.zernikeArray(NZERNS+1, config.sim.pupilSize)
piston = Zs[0]
Zs = Zs[1:]
Zs.shape = NZERNS, config.sim.pupilSize**2

# Test the large screen travelling over the telescope
# Buffer to store zernike coeffs
zCoeffs = numpy.zeros((NR0s, NIters*RUNS, NZERNS))

sCoord = int(round((config.sim.scrnSize-config.sim.pupilSize)/2)) # Coordinate to get middle of phase screen
for ir0, r0 in enumerate(R0s):
    print("Do Run with r0: {}".format(r0))
    config.atmos.r0 = r0
    for irun in range(RUNS):
        atmos = atmosphere.atmos(config.sim, config.atmos)
        for i in range(NIters):
            scrn = atmos.moveScrns()[0]
            subScrn = scrn[sCoord:-sCoord, sCoord:-sCoord] * (2*numpy.pi/500.)
            subScrn = subScrn.reshape(config.sim.pupilSize**2)
            # Get z coeffs
            zCoeffs[ir0, irun*NIters + i] = (Zs*subScrn).sum(1)/piston.sum()

zVars = zCoeffs.var(1)

# Load the noll reference values
noll = fits.getdata("resources/noll.fits").diagonal()[:NZERNS]

pyplot.figure()
pyplot.semilogy(noll, label="Noll")
for ir0, r0 in enumerate(R0s):
    pyplot.semilogy(zVars[ir0], label="r0:{}".format(r0))
pyplot.legend()

plt = bplot.figure()
plt.line(R0s, measuredR0s, legend="Measured")
plt.line(R0s, R0s, color="k", legend="Theoretical")
bplot.show(plt)

zVars.shape

zCoeffs.shape

# Test random screen generation
# Buffer to store zernike coeffs
zCoeffs = numpy.zeros((NR0s, NIters*RUNS, NZERNS))
config.atmos.randomScrns = True
sCoord = int(round((config.sim.scrnSize-config.sim.pupilSize)/2)) # Coordinate to get middle of phase screen
for ir0, r0 in enumerate(R0s):
    print("Do Run with r0: {}".format(r0))
    config.atmos.r0 = r0
    atmos = atmosphere.atmos(config.sim, config.atmos)
    for i in range(NIters*RUNS):
        scrn = atmos.moveScrns()[0]
        subScrn = scrn[sCoord:-sCoord, sCoord:-sCoord] * (2*numpy.pi/500.)
        subScrn = subScrn.reshape(config.sim.pupilSize**2)
        # Get z coeffs
        zCoeffs[ir0, i] = (Zs*subScrn).sum(1)/piston.sum()

zVars = zCoeffs.var(1)

pyplot.figure()
pyplot.semilogy(noll, label="Noll")
for ir0, r0 in enumerate(R0s):
    pyplot.semilogy(zVars[ir0], label="r0:{}".format(r0))
pyplot.legend()

measuredR0s = config.tel.telDiam/ ((zVars[:,2:]/noll[2:]).mean(1))**(3/5.)

plt = bplot.figure()
plt.line(R0s, measuredR0s, legend="Measured")
plt.line(R0s, R0s, color="k", legend="Theoretical")
bplot.show(plt)



