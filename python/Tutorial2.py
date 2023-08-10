get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import noisyduck as nd

# Geometry
res = 50 ; ri = 0.4 ; ro = 1.0
r = np.linspace(ri,ro,res)

# Define mean state
GAMMA=0.2; gam=1.4; vr=0.; vt=GAMMA/r; vz=0.3
p = (1./gam) + (GAMMA*GAMMA/2.)*(1. - 1./(r*r))

# Homentropic density: consistent with Kousen, Nijbour
rho = (1. + GAMMA*GAMMA*(gam-1.)*(1. - 1./(r*r))/2.)**(1./(gam-1.))

# Define circumferential and temporal wavenumber
omega=-10.; m=2

evals_r, evecs_rl, evecs_rr = nd.annulus.numerical.decomposition(omega,m,r,rho,vr,vt,vz,p,gam,filter='None',perturb_omega=True)
evals_f, evecs_fl, evecs_fr = nd.annulus.numerical.decomposition(omega,m,r,rho,vr,vt,vz,p,gam,filter='acoustic',alpha=0.00001,perturb_omega=True)

rho_evecs = evecs_fr[0*res:1*res,:]
vr_evecs  = evecs_fr[1*res:2*res,:]
vt_evecs  = evecs_fr[2*res:3*res,:]
vz_evecs  = evecs_fr[3*res:4*res,:]
p_evecs   = evecs_fr[4*res:5*res,:]

eigenvalues_nijbour = np.array([[-3.027340779492441,  48.36867862969005  ],
                                [-3.061040170507461,  42.82218597063621  ],
                                [-3.062863166930221,  37.11256117455139  ],
                                [-3.0646340777409034, 31.56606851549755  ],
                                [-3.066509159775743,  25.693311582381725 ],
                                [-3.0683842418105804, 19.8205546492659   ],
                                [-3.0703634950695786, 13.621533442088094 ],
                                [-3.008902472816523,  6.117455138662315  ],
                                [-3.0128609793345174,-6.280587275693314  ],
                                [-3.0471333120824085,-13.621533442088094 ],
                                [-3.0810931311578234,-19.9836867862969   ],
                                [-3.0830202988047404,-26.019575856443723 ],
                                [-3.0848432952275022,-31.729200652528547 ],
                                [-3.054737811445923, -37.438825448613386 ],
                                [-3.056508722256604, -42.98531810766721  ],
                                [-3.058331718679364, -48.694942903752036 ],
                                [-9.939361930417792, -0.08156606851550663],
                                [-12.940639069625966,-0.08156606851550663],
                                [3.9814554386754395, -0.08156606851550663],
                                [6.695376256044533,  -0.08156606851550663]])

fig = plt.figure()
# Plot raw numerical eigenvalues
for i in range(len(evals_r)):
    if (evals_r[i].imag > 0.):
        plt.plot(evals_r[i].real,evals_r[i].imag, 'b^',markersize=3)
    elif (evals_r[i].imag < 0.):
        plt.plot(evals_r[i].real,evals_r[i].imag, 'bs',markersize=3)
l = plt.xlabel('Eigenvalue: real')
l = plt.ylabel('Eigenvalue: imag')

# Plot filtered numerical eigenvalues
for i in range(len(evals_f)):
    if (evals_f[i].imag > 0.):
        h_up, = plt.plot(evals_f[i].real,evals_f[i].imag,   'b^',markersize=3, label='Acc. Down')
    elif (evals_f[i].imag < 0.):
        h_down, = plt.plot(evals_f[i].real,evals_f[i].imag, 'bs',markersize=3, label='Acc. Up'  )
# Plot analytical eigenvalues
h_analytical, = plt.plot(eigenvalues_nijbour[:,0],eigenvalues_nijbour[:,1], 'ko', markerfacecolor='None',markersize=7,label='Nijbour(2001)')
l=plt.legend(handles=[h_analytical,h_up,h_down],numpoints=1)
l = plt.xlabel('Eigenvalue: real')
l = plt.ylabel('Eigenvalue: imag')

# Plot first 4 eigenvectors of pressure, normalized by maximum value
fig = plt.figure(figsize=(4,5))
for i in range(4):
    loc = np.argmax(np.abs(p_evecs[:,2*i]))
    plt.plot(p_evecs[:,2*i].real/np.max(np.abs(p_evecs[loc,2*i].real)),r, 'ko')
l=plt.xlim((-1,1))
l=plt.ylim((0.4,1.0))
l=plt.xlabel('$Re(p_{mn})$')
l=plt.ylabel('Radius')

