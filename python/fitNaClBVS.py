# Initialize plotting within notebook.
get_ipython().magic('matplotlib inline')

from matplotlib.pyplot import *
rc('figure', figsize=(8, 6))

# DiffPy-CMI functions for loading data and building a fitting recipe
from diffpy.Structure import loadStructure
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.fitbase import FitRecipe, FitResults

# A least squares fitting algorithm from scipy
from scipy.optimize import leastsq

# Files containing our experimental data and structure file
dataFile = "NaCl.gr"
structureFile = "NaCl.cif"
spaceGroup = "F m -3 m"

# The first thing to construct is a contribution object which associates
# observed data and numerical model.  PDFContribution is a specialized
# contribution designed for PDF refinement of structure models.
# Here we create a new PDFContribution named "cpdf".
cpdf = PDFContribution("cpdf")

# Load the PDF data and set the r-range over which we'll fit.
cpdf.loadData(dataFile)
cpdf.setCalculationRange(xmin=1, xmax=30, dx=0.02)

# Add a structure model that will be used for PDF calculation.
nacl = loadStructure(structureFile)
cpdf.addStructure("nacl", nacl);

# Now cpdf.nacl now handles parameters for PDF calculation.
# cpdf.nacl.phase contains parameters relevant for the structure model.
# We can use the srfit function constrainAsSpaceGroup to constrain
# the lattice and ADP according to the relevant space group.
sgpars = constrainAsSpaceGroup(cpdf.nacl.phase, spaceGroup)
print("Space group parameters are " + ", ".join(p.name for p in sgpars) + ".")

# cpdf.nacl.phase also provides a restrainBVS function, which defines
# a soft restraint for agreement between the expected and calculated valences.
# restrainBVS returns the active Restraint object.  We save it so we can
# later manipulate its weight in the cost function.
rbv = cpdf.nacl.phase.restrainBVS()

# The FitRecipe does the work of managing refined variables and calculating
# residuals from all contributions and restraints.
thefit = FitRecipe()
# Turn off printing of iteration number.
thefit.clearFitHooks()

# We give our PDF model to the fit to be optimized.
thefit.addContribution(cpdf)

# We now link various model parameters to the fit variables that
# will be refined.  We will start with PDF scale, resolution damping
# factor qdamp and the peak sharpening coefficient delta2.
thefit.addVar(cpdf.scale, value=1)
thefit.addVar(cpdf.qdamp, value=0.03)
thefit.addVar(cpdf.nacl.delta2, value=5)

# We will also refine independent structure parameters that were found
# for our space group and atom coordinates.
for par in sgpars.latpars:
    thefit.addVar(par)
# Here we set the initial value for the anisotropic displacement
# parameters, because CIF had no ADP data.
for par in sgpars.adppars:
    thefit.addVar(par, value=0.005)
# Position parameters can be also constrained.  This does nothing
# for NaCl, because all atoms are at a special positions.
for par in sgpars.xyzpars:
    thefit.addVar(par)

# We can now execute the fit using scipy's least square optimizer.
# Let's define a few functions so it is easier to rerun the fit later.

def namesandvalues():
    "Format names and values of the active fit variables."
    return ' '.join("%s=%g" % nv for nv in zip(thefit.names, thefit.values))

def chattyfit():
    print("Refine PDF using scipy's least-squares optimizer:")
    print("  initial: " + namesandvalues())
    rv = leastsq(thefit.residual, thefit.values)
    print("  final: " +  namesandvalues())
    print('')
    return rv

def plotthefit():
    # Get the experimental data from the recipe
    r = thefit.cpdf.profile.x
    gobs = thefit.cpdf.profile.y
    gcalc = thefit.cpdf.evaluate()
    baseline = 1.1 * gobs.min()
    gdiff = gobs - gcalc
    figure()
    plot(r, gobs, 'bo', label="G(r) data",
        markerfacecolor='none', markeredgecolor='b')
    plot(r, gcalc, 'r-', label="G(r) fit")
    plot(r, gdiff + baseline, 'g-', label="G(r) diff")
    plot(r, 0.0 * r + baseline, 'k:')
    xlim(0, 30)
    xlabel(u"r (Å)")
    ylabel(u"G (Å$^{-2}$)")
    legend()
    return

# Perform the fit and plot it now.
chattyfit();
plotthefit();

# Report fit results:
results = FitResults(thefit)
print(results)

thefit.fix('delta2', 'U11_0', 'U11_4', 'qdamp')
thefit.a = 4
chattyfit()
plotthefit()

# Increase weight of the BVS restraint and refine again
rbv.sig = 0.1
chattyfit()
plotthefit()

import numpy
sigvalues = numpy.logspace(-4, 0)
avalues = []
for rbv.sig in sigvalues:
    leastsq(thefit.residual, thefit.values)
    avalues.append(thefit.a.value)

semilogx(sigvalues, avalues)
xlabel('sig parameter of the BVS restraint')
ylabel(u'cell parameter a (Å)');

