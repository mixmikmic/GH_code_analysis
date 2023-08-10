from IPython.display import Image
Image('../images/MarveliasGrossman2003a.png')

get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('../STN')
from STN import STN

# create an empty STN instance
stn = STN()

# enter any states where there are finite capacity constraints, 
# initial inventory, or with an associated price. In this problem
# we're trying to maximize the production of product B, so we
# an arbitrary value for B, and no price for any other state.

# states with an initial inventory
stn.state('A', init = 100)

# price data for product states
stn.state('B', price = 10)

# state -> task arcs
stn.stArc('A',  'Heat')
stn.stArc('hA', 'R1')
stn.stArc('hA', 'R2')
stn.stArc('IB', 'Sep')

# task -> state arcs with durations
stn.tsArc('Heat', 'hA', dur = 1)
stn.tsArc('R1',   'IB', dur = 3)
stn.tsArc('R2',   'IB', dur = 1)
stn.tsArc('Sep',  'B',  dur = 2)

# unit/task data
stn.unit('Heater',   'Heat', Bmax = 10)
stn.unit('Reactor1', 'R1',   Bmax =  4)
stn.unit('Reactor2', 'R2',   Bmax =  2)
stn.unit('Filter',   'Sep',  Bmax = 10)

# choose a time horizon, then build and solve model
H = 6
stn.build(range(0,H+1))
stn.solve()

stn.gantt()

stn.trace()



