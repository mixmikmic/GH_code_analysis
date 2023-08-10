import phoebe
from phoebe import u # units
import numpy as np
import matplotlib.pyplot as plt

logger = phoebe.logger()

b = phoebe.default_binary()

b.filter(context='constraint')

b['constraint']['primary']['mass']

print b.get_value('mass@primary@component')

print b['mass@primary@component']

b['asini@constraint']

b['esinw@constraint']

b['ecosw@constraint']

b['t0_perpass@constraint']

b['freq@constraint']

b['freq@binary@constraint']

b['freq@primary@constraint']

b['mass@constraint']

b['mass@primary@constraint']

b['sma@constraint']

b['sma@primary@constraint']

b['pot@constraint']

b['pot@primary@constraint']

b['period@constraint']

b['period@primary@constraint']

b['incl@constraint']

b['incl@primary@constraint']

print b['mass@primary@component'].constrained_by

print b['value@mass@primary@component'], b['value@mass@secondary@component'], b['value@period@orbit@component']

b.flip_constraint('mass@primary', 'period')

b['mass@primary@component'] = 1.0

print b['value@mass@primary@component'], b['value@mass@secondary@component'], b['value@period@orbit@component']

print b['constraint']

b['period@constraint@binary']

b['period@constraint@binary'].meta

b.flip_constraint('period@binary', 'mass')

b.set_value('q', 0.8)

print "M1: {}, M2: {}".format(b.get_value('mass@primary@component'),
                              b.get_value('mass@secondary@component'))

b.set_hierarchy('orbit:binary(star:secondary, star:primary)')

print b.get_value('q')

print "M1: {}, M2: {}".format(b.get_value('mass@primary@component'),
                              b.get_value('mass@secondary@component'))

print "M1: {}, M2: {}, period: {}, q: {}".format(b.get_value('mass@primary@component'),
                                                 b.get_value('mass@secondary@component'),
                                                 b.get_value('period@binary@component'),
                                                 b.get_value('q@binary@component'))

b.flip_constraint('mass@secondary@constraint', 'period')

print "M1: {}, M2: {}, period: {}, q: {}".format(b.get_value('mass@primary@component'),
                                                 b.get_value('mass@secondary@component'),
                                                 b.get_value('period@binary@component'),
                                                 b.get_value('q@binary@component'))

b.set_value('mass@secondary@component', 1.0)

print "M1: {}, M2: {}, period: {}, q: {}".format(b.get_value('mass@primary@component'),
                                                 b.get_value('mass@secondary@component'),
                                                 b.get_value('period@binary@component'),
                                                 b.get_value('q@binary@component'))

