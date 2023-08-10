import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.Bundle()

b = phoebe.Bundle.default_binary()

b = phoebe.default_binary()

print b.hierarchy

b = phoebe.default_binary(contact_binary=True)

print b.hierarchy

b = phoebe.Bundle()

b.add_component(phoebe.component.star, component='primary')
b.add_component('star', component='secondary')

b.add_star('extrastarforfun', teff=6000)

b.add_orbit('binary')

b.set_hierarchy(phoebe.hierarchy.binaryorbit, b['binary'], b['primary'], b['secondary'])

b.set_hierarchy(phoebe.hierarchy.binaryorbit(b['binary'], b['primary'], b['secondary']))

b.get_hierarchy()

b.set_hierarchy('orbit:binary(star:primary, star:secondary)')

b['hierarchy@system']

b.get_hierarchy()

b.hierarchy

print b.hierarchy.get_stars()

print b.hierarchy.get_orbits()

print b.hierarchy.get_top()

print b.hierarchy.get_parent_of('primary')

print b.hierarchy.get_children_of('binary')

print b.hierarchy.get_child_of('binary', 0)  # here 0 means primary component, 1 means secondary

print b.hierarchy.get_sibling_of('primary')

print b.hierarchy.get_primary_or_secondary('secondary')

