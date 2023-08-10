import numpy as np
import matplotlib.pyplot as plt

res = 100

y_gauss_p = np.zeros((res, 1))
y_gauss_n = np.zeros((res, 1))

for i, x in enumerate(np.linspace(-1, 1, num=res)):
    y_gauss_p[i] = np.e **(-x*x)
    y_gauss_n[i] = - np.e **(-x*x)
    

omag      = lambda x: 10**np.floor(np.log10(np.abs(x)))
signifFig = lambda x, n: (np.around(x/omag(x),n)*omag(x))

sigfigs = 9

r = 0
r_low = 0.001
r_up = 1

done = 0
counter = 0

cir_up = np.zeros((res, 1)) 
cir_down = np.zeros ((res, 1))

while done == 0:
    
    r = r_low + ((r_up - r_low) / 2)
    #print(str(r_up) + '   ' + str(r_low) + '   ' + str(r))
    for i, x in enumerate(np.linspace(-1, 1, num=res)):
        cir_up[i] = np.sqrt((r**2) - (x**2))
        cir_down[i] = -np.sqrt((r**2) - (x**2))
    
    diff = np.subtract(y_gauss_p,cir_up)
    
    if any( diff < 0 ):
        r_up = r
    else:
        r_low = r
    
    if 0 in np.subtract(signifFig(cir_up,sigfigs), signifFig(y_gauss_p,sigfigs)):
        done = 1
        print(r)
        break
    
plt.plot(np.linspace(-1, 1, num=res), y_gauss_p)
plt.plot(np.linspace(-1, 1, num=res), y_gauss_n)
plt.plot(np.linspace(-1, 1, num=res), cir_up)
plt.plot(np.linspace(-1, 1, num=res), cir_down)
plt.show()

print(r)



