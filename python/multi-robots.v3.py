# necessary stuff to set the paths:
import sys
sys.path.append("..")

# Import and create the connection to the simulator:
from vrep.simulator import Simulator
simulator = Simulator()

epuck = simulator.get_epuck()

simulator.start()

epuck.rot_spd = 0.2

another_epuck = simulator.get_epuck("#0")

another_epuck.rot_spd = -0.2

simulator.robots

simulator.robots[0].rot_spd = 0.5

epuck.rot_spd = 0.5

for robot in simulator.robots:
    robot.rot_spd = 2.



