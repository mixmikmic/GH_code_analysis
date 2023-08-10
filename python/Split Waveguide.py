from pypropagate import *
get_ipython().magic('matplotlib inline')

settings = presets.settings.create_paraxial_wave_equation_settings()

sw = settings.create_category('split_waveguide',short_name='SW')
sw.create_symbol('l_1',info='propagation length before split')
sw.create_symbol('alpha',info='separation angle')
sw.create_symbol('d',info='waveguide profile edge length')
sw.create_symbol('n_core',info='refractive index in the guiding core')
sw.create_symbol('n_cladding',info='refractive index in the cladding');

s = settings.symbols

settings.wave_equation.n = pc.piecewise(
    (sw.n_core,(abs(s.x)<sw.d/2) & (abs(s.y)<sw.d/2) & (s.z <= sw.l_1)), # initial channel
    (sw.n_core,(abs(s.x - pc.tan(sw.alpha) * (s.z - sw.l_1) ) < sw.d/2) & (abs(s.y)<sw.d/2) & (s.z > sw.l_1)), # upper channel
    (sw.n_core,(abs(s.x + pc.tan(sw.alpha) * (s.z - sw.l_1) ) < sw.d/2) & (abs(s.y)<sw.d/2) & (s.z > sw.l_1)), # lower channel
    (sw.n_cladding,True) # cladding
)

settings.get(settings.wave_equation.n)

settings.wave_equation.set_energy(12*units.keV)
presets.boundaries.set_plane_wave_initial_conditions(settings)

sw.n_core = 1
sw.n_cladding = presets.medium.create_material('Ti',settings)
sw.l_1 = 0.2 * units.mm
sw.d = 70 * units.nm
sw.alpha = 0.01 * units.degrees

settings.simulation_box.set((0.5*units.um,0.3*units.um,1*units.mm),(1000,500,1000))
plot(s.n.subs(s.y,0),settings,figsize=(10,4))
plot(s.n.subs(s.z,s.zmax),settings,figsize=(3,4));

settings.simulation_box.set((0.5*units.um,0.3*units.um,1*units.mm),(1000,500,5000))

propagator = propagators.FiniteDifferences3D(settings)
plot(propagator.run_slice()[-0.2*units.um:0.2*units.um,0,::1*units.um],figsize=(16,5));

plot(propagator.get_field(),figsize=(5,7));

