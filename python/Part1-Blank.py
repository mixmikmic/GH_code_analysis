import openmc

#TODO: Define the uo2 Material here.

#TODO: Add 0.03 of 'U235' and 0.97 of 'U238'.
#TODO: Add 2.0 of 'O' and set the density to 10.0 'g/cm3'.

zirconium = openmc.Material()
zirconium.add_element('Zr', 1.0)
zirconium.set_density('g/cm3', 6.6)

water = openmc.Material()
water.add_element('H', 2.0)
water.add_element('O', 1.0)
water.set_density('g/cm3', 0.7)

#TODO: Add an s_alpha_beta table for 'c_H_in_H2O'

#TODO: Define Materials and export_to_xml().

#TODO: Define fuel_or with R=0.39 cm.
clad_ir = openmc.ZCylinder(R=0.40)
clad_or = openmc.ZCylinder(R=0.46)

#TODO: Define a pitch of 1.26 cm.
#TODO: Define the left XPlane.
right = openmc.XPlane(x0=pitch/2, boundary_type='reflective')
bottom = openmc.YPlane(y0=-pitch/2, boundary_type='reflective')
top = openmc.YPlane(y0=pitch/2, boundary_type='reflective')

#TODO: Define the fuel_region and gap_region.
clad_region = +clad_ir & -clad_or
water_region = +left & -right & +bottom & -top & +clad_or

#TODO: Define fuel and gap Cell objects.

clad = openmc.Cell()
clad.fill = zirconium
clad.region = clad_region

moderator = openmc.Cell()
moderator.fill = water
moderator.region = water_region

#TODO: Define the root Universe.

g = openmc.Geometry()
g.root_universe = root
g.export_to_xml()
get_ipython().system('cat geometry.xml')

#TODO: Define the Plot p.

#TODO: Give it a width of [pitch, pitch].
#TODO: Give it [400, 400] pixels.
#TODO: Set color_by to 'material'
p.colors = {uo2:'salmon', water:'cyan', zirconium:'gray'}

openmc.plot_inline(p)

point = openmc.stats.Point((0, 0, 0))
src = openmc.Source(space=point)

#TODO: Create a Settings object.

#TODO: Assign the source.
#TODO: Set 100 batches, 10 inactive, 1000 particles.

settings.export_to_xml()
get_ipython().system('cat settings.xml')

#TODO: Define a Tally t with name='fuel tally'

#TODO: Make a CellFilter for the fuel cell.
#TODO: Set the tally filters to [cell_filter].

t.nuclides = ['U235']
t.scores = ['total', 'fission', 'absorption', '(n,gamma)']

tallies = openmc.Tallies([t])
tallies.export_to_xml()
get_ipython().system('cat tallies.xml')

#TODO: Run openmc.

get_ipython().system('cat tallies.out')

