import numpy as np
import matplotlib.pyplot as plt

import slr

sliceThickness = 5.
z = np.linspace(-3*sliceThickness,3*sliceThickness,400)
numberPoints = 400

tbwSLR = 3.55
durationSLR = 3.1
rfSLR = slr.slr("se",numberPoints,tbwSLR,durationSLR,flipAngle=160,filterType="ls")
rfSLR.GenerateRF()
rfScaledSLR = rfSLR.GetRFScaled()
mxySLR = rfSLR.Simulate(sliceThickness,z)
timeSLR = np.linspace(0,durationSLR,numberPoints)

tbwMsinc = 3.55
durationMsinc = 1.2
rfMsinc = slr.slr("smalltip",numberPoints,tbwMsinc,durationMsinc,flipAngle=160.)
rfMsinc.GenerateRF()
rfScaledMsinc = rfMsinc.GetRFScaled()
mxyMsinc = rfMsinc.Simulate(sliceThickness,z,simulationType="se")
timeMsinc = np.linspace(0,durationMsinc,numberPoints)

tbwMP = 3.55
durationMP = 3.1
rfMP = slr.slr("se",numberPoints,tbwMP,durationMP,flipAngle=160.,filterType="min")
rfMP.GenerateRF()
rfScaledMP = rfMP.GetRFScaled()
mxyMP = rfMP.Simulate(sliceThickness,z,simulationType="se")
timeMP = np.linspace(0,durationMP,numberPoints)

plt.figure()
p1, = plt.plot(timeMsinc,rfScaledMsinc.real)
p2, = plt.plot(timeSLR,rfScaledSLR.real)
p3, = plt.plot(timeMP,rfScaledMP.real)
plt.xlabel("time [ms]")
plt.ylabel("B1 [G]")
plt.legend([p1,p2,p3],["m-sinc","ls SLR","min phase SLR"])

plt.figure()
p1, = plt.plot(z,abs(mxyMsinc))
p2, = plt.plot(z,abs(mxySLR))
p3, = plt.plot(z,abs(mxyMP))
plt.xlabel("slice positiion [mm]")
plt.ylabel("transverse magnetization Mxy")
plt.legend([p1,p2,p3],["m-sinc","SLR","min phase SLR"])
plt.show()



