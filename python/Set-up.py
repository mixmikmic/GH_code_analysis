##On CLI under sudo
#!pip3 install 'git+https://github.com/Troxid/vrep-api-python'

import os

if "VREP_VM" in os.environ:
    os.environ["VREP"]='/opt/V-REP_PRO_EDU_V3_4_0_Linux'
    os.environ["VREP_LIBRARY"]=os.environ["VREP"]+'/programming/remoteApiBindings/lib/lib/64Bit/'
else:
    os.environ["VREP"]='/Applications/V-REP_PRO_EDU_V3_4_0_Mac'
    os.environ["VREP_LIBRARY"]=os.environ["VREP"]+'/programming/remoteApiBindings/lib/lib/'

def show_methods(c):
    if type(c) != type: c=type(c)
    methods = [method for method in dir(c) if not method.startswith('_')]
    print('Methods available in {}:\n\t{}'.format(c.__name__ , '\n\t'.join(methods)))

def show_attributes(r):
    print('State elements for the {}:\n\t{}'.format(type(r).__name__ ,
                                                    '\n\t'.join(list(vars(r).keys()))))

from pyrep import VRep
from pyrep.vrep import vrep as vrep

from __future__ import print_function
from IPython.core.magic import (Magics, magics_class, line_magic,
                                cell_magic, line_cell_magic)
import shlex

# The class MUST call this class decorator at creation time
@magics_class
class Vrep_Sim(Magics):

    @cell_magic
    def vrepsim(self, line, cell):
        "V-REP magic"
        
        args=shlex.split(line)
        
        if len(args)>1:
            #Use shlex.split to handle quoted strings containing a space character
            loadSceneRelativeToClient(args[0])
            #Get the robot class from the string
            robotclass=eval(args[1])
        else:
            #Get the robot class from the string
            robotclass=eval(args[0])
        
        #Handle default IP address and port settings; grab from globals if set
        ip = self.shell.user_ns['vrep_ip'] if 'vrep_ip' in self.shell.user_ns else '127.0.0.1'
        port = self.shell.user_ns['vrep_port'] if 'vrep_port' in self.shell.user_ns else 19997
        
        #The try/except block exits form a keyboard interrupt cleanly
        try:
            #Create a connection to the simulator
            with VRep.connect(ip, port) as api:
                #Set the robot variable to an instance of the desired robot class
                robot = robotclass(api)
                #Execute the cell code - define robot commands as calls on: robot
                exec(cell)
        except KeyboardInterrupt:
            pass

    #@line_cell_magic
    @line_magic
    def vrep_robot_methods(self, line):
        "Show methods"
        robotclass = eval(line)
        methods = [method for method in dir(robotclass) if not method.startswith('_')]
        print('Methods available in {}:\n\t{}'.format(robotclass.__name__ , '\n\t'.join(methods)))

#Could install as magic separately
ip = get_ipython()
ip.register_magics(Vrep_Sim)

