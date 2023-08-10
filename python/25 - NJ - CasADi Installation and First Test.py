from sys import path
path.append(r"../../lib/casadi-py27-np1.9.1-v3.0.0")
from casadi import *

x = MX.sym("x")
print jacobian(sin(x),x)



