import math
import cmath
import numpy as np

import kali.carma

QPOTask = kali.carma.CARMATask(2,1)
dt = 0.1
Rho = np.array([complex(-1.0/100.0, (2.0*math.pi)/50.0), complex(-1.0/100.0, -(2.0*math.pi)/50.0), -1.0/10.0, 1.0])
Theta = kali.carma.coeffs(2, 1, Rho)
print Theta
QPOTask.check(Theta)
QPOTask.set(dt, Theta)
QPOLC = QPOTask.simulate(duration=2000.0)
QPOLC.plot()

QPOTask.plotpsd()

QPOTask.plotsf()

QPOTask2 = kali.carma.CARMATask(4,1)
dt = 0.1
Rho2 = np.array([complex(-1.0/100.0, (2.0*math.pi)/50.0), complex(-1.0/100.0, -(2.0*math.pi)/50.0), complex(-1.0/400.0, (2.0*math.pi)/250.0), complex(-1.0/400.0, -(2.0*math.pi)/250.0), -1.0/10.0, 1.0])
Theta2 = kali.carma.coeffs(4, 1, Rho2)
print Theta2
QPOTask2.check(Theta2)
QPOTask2.set(dt, Theta2)
QPOLC2 = QPOTask2.simulate(duration=2000.0)
QPOLC2.plot()

QPOTask2.plotpsd()

QPOTask2.plotsf()

