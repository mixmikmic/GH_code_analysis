from math import pi

get_ipython().magic('run matplotlib_setup.ipy')
from matplotlib import pyplot
import numpy as np
import kwant

lat=kwant.lattice.square()
L,W=30,16
def myshape(R): return ( 
        (R[0]**2 + R[1]**2) > (L-W/2)**2 and                 
        (R[0]**2 + R[1]**2) < (L+W/2)**2)

#L,W=30,16
#def myshape_ellipse(R): return ( 
#        (R[0]**2/2. + R[1]**2) > (L-W/2)**2 and                 
#        (R[0]**2/2. + R[1]**2) < (L+W/2)**2)


H=kwant.Builder()

H[lat.shape(myshape,(L,0) )]=4

#H[lat.shape(myshape_ellipse,(int(L*1.14),0)) ]=4


H[lat.neighbors()]=1

def Aharonov_Bohm(site1,site2,phi): return np.exp(-2j*pi*phi)
    
for hop in H.hoppings():
    if hop[0].tag[0]==1 and hop[0].tag[1]>0 and hop[1].tag[0]==0: 
        H[hop]=Aharonov_Bohm

sym=kwant.TranslationalSymmetry(lat.vec((1,0)))
def lead_shape(R): return abs(R[1]) < W/2 and abs(R[0]) <3
Hlead =kwant.Builder(sym)
Hlead[lat.shape(lead_shape,(0,0) )]=4
Hlead[lat.neighbors()]=1
H.attach_lead(Hlead)
H.attach_lead(Hlead.reversed())
kwant.plot(H);


Hf=H.finalized()
data = []
phis = np.linspace(0,1.,50)
for phi in phis:
    smatrix = kwant.smatrix(Hf, 3.3,args=[phi])
    data.append(smatrix.transmission(1, 0))
pyplot.plot(phis, data,'o');
pyplot.xlabel('$\phi = BS/(h/e)$')
pyplot.ylabel('g in unit of $(2e^2/h)$');
pyplot.title('Aharonov-Effect')



get_ipython().magic('matplotlib inline')
import numpy as np
from math import pi
from matplotlib import pyplot
import kwant

lat=kwant.lattice.square()
L,W=100,12

def myshape(R): return ( 
        (R[0]**2 + R[1]**2) > (L-W/2)**2 and                 
        (R[0]**2 + R[1]**2) < (L+W/2)**2)
H=kwant.Builder()
H[lat.shape(myshape,(L,0) )]=4

def Field(site1,site2,phi):
    x1,y1=site1.pos
    x2,y2=site2.pos
    return -np.exp(-0.5j * phi * (x1 - x2) * (y1 + y2))

H[lat.neighbors()] = Field

sym=kwant.TranslationalSymmetry(lat.vec((1,0)))
def lead_shape(R): return abs(R[1]) < W/2 and abs(R[0]) <3
Hlead =kwant.Builder(sym)
Hlead[lat.shape(lead_shape,(0,0) )]=4
Hlead[lat.neighbors()]=Field
H.attach_lead(Hlead)
H.attach_lead(Hlead.reversed())
kwant.plot(H);

Hf=H.finalized()
data = []
phis = np.linspace(0.,0.0005,50)
for phi in phis:
    smatrix = kwant.smatrix(Hf, 3.3,args=[phi])
    data.append(smatrix.transmission(1, 0))
pyplot.plot(phis, data);
pyplot.xlabel('$\phi = Ba^2/(h/e)$')
pyplot.ylabel('g in unit of $(2e^2/h)$');
pyplot.title('Aharonov-Effect')

