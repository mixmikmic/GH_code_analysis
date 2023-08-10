import numpy as np
import matplotlib.pyplot as plt
r2 = np.linspace(0.0123,0.02,10) # vector for range of r2 values
Q = np.empty([10,1]) # empty output vector for heat transfer values 

#Assumptions: T1 is max temp throughout chamber, T2 is ~Curie point of thermocouple divided by the FOS (1.4) 
kc = 11.4 #[W/mK] thermal conductivity of Inconel 718
L = .01778 #[m] length of combustion chamber
r1 = .0122 #[m] combustion chamber radius
T1 = 3065.6 #[K] combustion chamber temp.
T2 = 450 #[K] surface temp.

Q = np.true_divide(kc*2*np.pi*L*(T1-T2),(np.log(r2/r1)))

plt.xkcd()
plt.plot(r2,Q)
plt.axis([0.012, .021, 0, 45000])
plt.xlabel('outer radius [m]') #this is where the thermocouple would be placed
plt.ylabel('Heat transferred through comb. chamber [W]') #energy transferred from combustion chamber to thermocouple via conduction
plt.show()

Q[list(r2).index(0.018288888888888889)]



