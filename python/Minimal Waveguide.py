from pypropagate import *
get_ipython().magic('matplotlib inline')

settings = presets.settings.create_paraxial_wave_equation_settings()
settings.simulation_box.set((0.25*units.um,0.25*units.um,0.25*units.mm),(1000,1000,1000))
presets.boundaries.set_plane_wave_initial_conditions(settings)
settings.wave_equation.set_energy(12*units.keV)

nVa = 1
nGe = presets.medium.create_material('Ge',settings)

s = settings.symbols
waveguide_radius = 25*units.nm
settings.wave_equation.n = pc.piecewise((nVa,pc.sqrt(s.x**2+s.y**2) <= waveguide_radius),(nGe,True))
settings.get_numeric(s.n)

propagator = propagators.FiniteDifferences2D(settings)

field = propagator.run_slice()[-2*waveguide_radius:2*waveguide_radius]
plot(field,figsize = (13,5));

propagator = propagators.FiniteDifferencesCS(settings)

field = propagator.run_slice()[-2*waveguide_radius:2*waveguide_radius]
plot(field,figsize = (13,5));

