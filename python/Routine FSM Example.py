# Setup the notebook   
from transitions.extensions import GraphMachine as Machine
from IPython.display import Image, display, display_png

from machines.routine import DailyRoutineMachine
from random import random

# GraphMachine = MachineFactory.get_predefined(graph=True, nested=True)

from machines.routine import *

machine = DailyRoutineMachine('machine')
machine.show_graph()  # The current state should be in orange.
machine.state

machine.go_to_bed()
machine.show_graph()
machine.state

machine.get_up()
machine.show_graph()

machine.yes()
machine.show_graph()



