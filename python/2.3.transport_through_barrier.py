import numpy as np
import kwant
get_ipython().magic('run matplotlib_setup.ipy')
from matplotlib import pyplot

lat = kwant.lattice.square()

def make_lead_x(W=10, t=1):
    syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
    syst[(lat(0, y) for y in range(W))] = 4 * t
    syst[lat.neighbors()] = -t
    return syst

def make_wire_with_flat_potential(W=10, L=2, t=1):
    def onsite(s, V):
        return (4 - V) * t

    # Construct the scattering region.
    sr = kwant.Builder()
    sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    sr[lat.neighbors()] = -t

    # Build and attach lead from both sides.
    lead = make_lead_x(W, t)
    sr.attach_lead(lead)
    sr.attach_lead(lead.reversed())

    return sr

def plot_transmission(syst, energy, params):
    # Compute conductance
    trans = []
    for param in params:
        smatrix = kwant.smatrix(syst, energy, args=[param])
        trans.append(smatrix.transmission(1, 0))
    pyplot.plot(params, trans)

_syst = make_wire_with_flat_potential()
kwant.plot(_syst)
_syst = _syst.finalized()
kwant.plotter.bands(_syst.leads[0])
plot_transmission(_syst, 1, np.linspace(-2, 0, 51))

from math import atan2, pi, sqrt

def rectangular_gate_pot(distance, left, right, bottom, top):
    """Compute the potential of a rectangular gate.
    
    The gate hovers at the given distance over the plane where the
    potential is evaluated.
    
    Based on J. Appl. Phys. 77, 4504 (1995)
    http://dx.doi.org/10.1063/1.359446
    """
    d, l, r, b, t = distance, left, right, bottom, top

    def g(u, v):
        return atan2(u * v, d * sqrt(u**2 + v**2 + d**2)) / (2 * pi)

    def func(x, y, voltage):
        return voltage * (g(x-l, y-b) + g(x-l, t-y) +
                          g(r-x, y-b) + g(r-x, t-y))

    return func

_gate1 = rectangular_gate_pot(10, 20, 50, -50, 15)
_gate2 = rectangular_gate_pot(10, 20, 50, 25, 90)
def qpc_potential(site, V):
    x, y = site.pos
    return _gate1(x, y, V) + _gate2(x, y, V)

def make_barrier(pot, W=40, L=70, t=1):
    def onsite(*args):
        return 4 * t - pot(*args)

    # Construct the scattering region.
    sr = kwant.Builder()
    sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    sr[lat.neighbors()] = -t

    # Build and attach lead from both sides.
    lead = make_lead_x(W, t)
    sr.attach_lead(lead)
    sr.attach_lead(lead.reversed())

    return sr

qpc = make_barrier(qpc_potential)
kwant.plot(qpc);
kwant.plotter.map(qpc, lambda s: qpc_potential(s, 1));

fqpc = qpc.finalized()
kwant.plotter.bands(fqpc.leads[0]);

plot_transmission(fqpc, 0.3, np.linspace(-1, 0, 101))

