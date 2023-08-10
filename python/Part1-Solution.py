import openmc

uo2 = openmc.Material()

uo2.add_nuclide('U235', 0.03)
uo2.add_nuclide('U238', 0.97)
uo2.add_element('O', 2.0)
uo2.set_density('g/cm3', 10.0)

zirconium = openmc.Material()
zirconium.add_element('Zr', 1.0)
zirconium.set_density('g/cm3', 6.6)

water = openmc.Material()
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.7)

water.add_s_alpha_beta('c_H_in_H2O')

mf = openmc.Materials((uo2, zirconium, water))
mf.export_to_xml()
get_ipython().system('cat materials.xml')

fuel_or = openmc.ZCylinder(R=0.39)
clad_ir = openmc.ZCylinder(R=0.40)
clad_or = openmc.ZCylinder(R=0.46)

pitch = 1.26
left = openmc.XPlane(x0=-pitch/2, boundary_type='reflective')
right = openmc.XPlane(x0=pitch/2, boundary_type='reflective')
bottom = openmc.YPlane(y0=-pitch/2, boundary_type='reflective')
top = openmc.YPlane(y0=pitch/2, boundary_type='reflective')

fuel_region = -fuel_or
gap_region = +fuel_or & -clad_ir
clad_region = +clad_ir & -clad_or
water_region = +left & -right & +bottom & -top & +clad_or

fuel = openmc.Cell()
fuel.fill = uo2
fuel.region = fuel_region

gap = openmc.Cell()
gap.fill = 'void'
gap.region = gap_region

clad = openmc.Cell()
clad.fill = zirconium
clad.region = clad_region

moderator = openmc.Cell()
moderator.fill = water
moderator.region = water_region

root = openmc.Universe()
root.add_cells((fuel, gap, clad, moderator))

g = openmc.Geometry()
g.root_universe = root
g.export_to_xml()
get_ipython().system('cat geometry.xml')

p = openmc.Plot()

p.width = [pitch, pitch]
p.pixels = [400, 400]
p.color_by = 'material'
p.colors = {uo2:'salmon', water:'cyan', zirconium:'gray'}

openmc.plot_inline(p)

point = openmc.stats.Point((0, 0, 0))
src = openmc.Source(space=point)

settings = openmc.Settings()

settings.source = src
settings.batches = 100
settings.inactive = 10
settings.particles = 1000

settings.export_to_xml()
get_ipython().system('cat settings.xml')

t = openmc.Tally(name='fuel tally')

cell_filter = openmc.CellFilter(fuel.id)
t.filters = [cell_filter]

t.nuclides = ['U235']
t.scores = ['total', 'fission', 'absorption', '(n,gamma)']

tallies = openmc.Tallies([t])
tallies.export_to_xml()
get_ipython().system('cat tallies.xml')

openmc.run()

get_ipython().system('cat tallies.out')

