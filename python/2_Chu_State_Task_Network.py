from IPython.display import Image
Image('../images/Chu_2013.png')

get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('../STN')
from STN import STN

# create instance
stn = STN()

# states
stn.state('M1', capacity = 500, init = 500)
stn.state('M2', capacity = 500, init = 500)
stn.state('M3', capacity = 500, init = 500)
stn.state('M3', capacity = 500, init = 500)
stn.state('M4', capacity = 500, init = 500)
stn.state('I1', capacity = 100)
stn.state('I2', capacity = 100)
stn.state('I3', capacity = 100)
stn.state('I4', capacity = 100)
stn.state('I5', capacity = 100)
stn.state('I6', capacity = 100)
stn.state('P1', capacity = 500)
stn.state('P2', capacity = 500)
stn.state('P3', capacity = 500)
stn.state('P4', capacity = 500)

# state to task arcs
stn.stArc('M1', 'Reaction_1', rho = 0.8)
stn.stArc('M2', 'RM Prep',    rho = 0.5)
stn.stArc('M3', 'RM Prep',    rho = 0.5)
stn.stArc('M4', 'Reaction_2', rho = 0.7)
stn.stArc('I1', 'Reaction_1', rho = 0.2)
stn.stArc('I1', 'Reaction_2', rho = 0.3)
stn.stArc('I1', 'Reaction_3', rho = 0.4)
stn.stArc('I2', 'Reaction_3', rho = 0.6)
stn.stArc('I3', 'Packing_1',  rho = 1.0)
stn.stArc('I4', 'Packing_2',  rho = 1.0)
stn.stArc('I5', 'Drum_1',     rho = 1.0)
stn.stArc('I6', 'Drum_2',     rho = 1.0)

# task to state arcs
stn.tsArc('RM Prep',    'I1', rho = 1.0, dur =  72)
stn.tsArc('Reaction_1', 'I3', rho = 1.0, dur = 162)
stn.tsArc('Reaction_2', 'I2', rho = 1.0, dur = 138)
stn.tsArc('Reaction_3', 'I4', rho = 1.0, dur = 162)
stn.tsArc('Packing_1',  'P1', rho = 0.5, dur = 108)
stn.tsArc('Packing_1',  'I5', rho = 0.5, dur = 108)
stn.tsArc('Packing_2',  'I6', rho = 0.5, dur = 108)
stn.tsArc('Packing_2',  'P4', rho = 0.5, dur = 108)
stn.tsArc('Drum_1',     'P2', rho = 1.0, dur =  90)
stn.tsArc('Drum_2',     'P3', rho = 1.0, dur =  90)

# unit-task information

stn.unit('RM Prep',   'RM Prep',    Bmax = 100, cost = 1000, vcost =  50)
stn.unit('Reactor_1', 'Reaction_1', Bmax =  80, cost = 3000, vcost = 250)
stn.unit('Reactor_1', 'Reaction_2', Bmax =  50, cost = 1500, vcost = 150)
stn.unit('Reactor_1', 'Reaction_3', Bmax =  80, cost = 2000, vcost = 100)
stn.unit('Reactor_2', 'Reaction_1', Bmax =  80, cost = 3000, vcost = 250)
stn.unit('Reactor_2', 'Reaction_2', Bmax =  50, cost = 1500, vcost = 150)
stn.unit('Reactor_2', 'Reaction_3', Bmax =  80, cost = 2000, vcost = 100)
stn.unit('Finishing', 'Packing_1',  Bmax = 100, cost =  500, vcost =  20)
stn.unit('Finishing', 'Packing_2',  Bmax = 100, cost =  500, vcost =  20)
stn.unit('Drumming',  'Drum_1',     Bmax =  50, cost =  200, vcost =  50)
stn.unit('Drumming',  'Drum_2',     Bmax =  50, cost =  200, vcost =  50)

N = 149
H = 6*N
stn.build(range(0,H+1,6))

# production constraints
stn.model.cons.add(stn.model.S['P1',H] == 100)
stn.model.cons.add(stn.model.S['P2',H] == 100)
stn.model.cons.add(stn.model.S['P3',H] ==  50)
stn.model.cons.add(stn.model.S['P4',H] ==  50)

stn.solve('gurobi')

stn.gantt()

N = 203
H = 6*N
stn.build(range(0,H+1,6))

# production constraints
stn.model.cons.add(stn.model.S['P1',H] == 200)
stn.model.cons.add(stn.model.S['P2',H] == 200)
stn.model.cons.add(stn.model.S['P3',H] == 100)
stn.model.cons.add(stn.model.S['P4',H] == 100)

stn.solve('gurobi')

stn.gantt()

# sequence dependent changever time from task1 to task2

stn.changeover('Reaction_1', 'Reaction_1', 12)
stn.changeover('Reaction_1', 'Reaction_2', 30)
stn.changeover('Reaction_1', 'Reaction_3', 30)
stn.changeover('Reaction_2', 'Reaction_1', 30)
stn.changeover('Reaction_2', 'Reaction_2', 12)
stn.changeover('Reaction_2', 'Reaction_3',  6)
stn.changeover('Reaction_3', 'Reaction_1', 30)
stn.changeover('Reaction_3', 'Reaction_2', 30)
stn.changeover('Reaction_3', 'Reaction_3', 12)

stn.changeover('Packing_1',  'Packing_1',   0)
stn.changeover('Packing_1',  'Packing_2',   6)
stn.changeover('Packing_2',  'Packing_1',   6)
stn.changeover('Packing_1',  'Packing_1',   0)

N = 215
H = 6*N
stn.build(range(0,H+1,6))

# production constraints
stn.model.cons.add(stn.model.S['P1',H] == 200)
stn.model.cons.add(stn.model.S['P2',H] == 200)
stn.model.cons.add(stn.model.S['P3',H] == 100)
stn.model.cons.add(stn.model.S['P4',H] == 100)

stn.solve('gurobi')

stn.gantt()



