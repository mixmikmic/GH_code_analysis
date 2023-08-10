get_ipython().magic('pylab inline')

import seaborn as sns

sns.set_context('poster', font_scale=1.25)

w=np.array([4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,    
            1./36.,1./36.,1./36.]) # weights for directions
cx=np.array([0,1,0,-1,0,1,-1,-1,1]) # direction vector for the x direction
cy=np.array([0,0,1,0,-1,1,1,-1,-1]) # direction vector for the y direction
tau=1
cs=1/math.sqrt(3)
cs2 = cs**2
cs22 = 2*cs2
cssq = 2.0/9.0

w0 = 4./9.
w1 = 1./9.
w2 = 1./36.

viscosity = cs**2*(tau-0.5)
print 'Viscosity is:' , viscosity

lx=400  # length of domain in the x direction
# extend by one to deal with domain boundaries at walls...
nx = lx + 1
ly=400  # length of domain in the y direction
ny = ly + 1

rho = np.ones((nx, ny))
u_applied=cs/100
u = u_applied*(np.ones((nx, ny)) + np.random.randn(nx,ny))
v= (u_applied/100.)*(np.ones((nx, ny)) + np.random.randn(nx,ny)) # initializing the vertical velocities

f=np.zeros((9,nx,ny)) # initializing f
feq = np.zeros((9, nx, ny))

# Taken from sauro succi's code. This will be super easy to put on the GPU.

def update_feq():
    ul = u/cs2
    vl = v/cs2
    uv = ul*vl
    usq = u*u
    vsq = v*v
    sumsq  = (usq+vsq)/cs22
    sumsq2 = sumsq*(1.-cs2)/cs2
    u2 = usq/cssq 
    v2 = vsq/cssq

    feq[0, :, :] = w0*(1. - sumsq)

    feq[1, :, :] = w1*(1. - sumsq  + u2 + ul)
    feq[2, :, :] = w1*(1. - sumsq  + v2 + vl)
    feq[3, :, :] = w1*(1. - sumsq  + u2 - ul)
    feq[4, :, :] = w1*(1. - sumsq  + v2 - vl)
    feq[5, :, :] = w2*(1. + sumsq2 + ul + vl + uv)
    feq[6, :, :] = w2*(1. + sumsq2 - ul + vl - uv)
    feq[7, :, :] = w2*(1. + sumsq2 - ul - vl + uv)
    feq[8, :, :] = w2*(1. + sumsq2 + ul - vl - uv)

update_feq()

f = feq.copy()
# We now slightly perturb f
amplitude = .01
perturb = (1. + amplitude*np.random.randn(nx, ny))
f *= perturb

def move_bcs():
    # West inlet: periodic BC's
    for j in range(1,ly):
        f[1,0,j] = f[1,lx,j]
        f[5,0,j] = f[5,lx,j]
        f[8,0,j] = f[8,lx,j]
    # EAST outlet
    for j in range(1,ly):
        f[3,lx,j] = f[3,0,j]
        f[6,lx,j] = f[6,0,j]
        f[7,lx,j] = f[7,0,j]
    # NORTH solid
    for i in range(1, lx): # Bounce back
        f[4,i,ly] = f[2,i,ly-1]
        f[8,i,ly] = f[6,i+1,ly-1]
        f[7,i,ly] = f[5,i-1,ly-1]
    # SOUTH solid
    for i in range(1, lx):
        f[2,i,0] = f[4,i,1]
        f[6,i,0] = f[8,i-1,1]
        f[5,i,0] = f[7,i+1,1]
        
    # Corners bounce-back
    f[8,0,ly] = f[6,1,ly-1]
    f[5,0,0]  = f[7,1,1]
    f[7,lx,ly] = f[5,lx-1,ly-1]
    f[6,lx,0]  = f[8,lx-1,1]

def move():
    for j in range(ly,0,-1): # Up, up-left
        for i in range(0, lx):
            f[2,i,j] = f[2,i,j-1]
            f[6,i,j] = f[6,i+1,j-1]
    for j in range(ly,0,-1): # Right, up-right
        for i in range(lx,0,-1):
            f[1,i,j] = f[1,i-1,j]
            f[5,i,j] = f[5,i-1,j-1]
    for j in range(0,ly): # Down, right-down
        for i in range(lx,0,-1):
            f[4,i,j] = f[4,i,j+1]
            f[8,i,j] = f[8,i-1,j+1]
    for j in range(0,ly): # Left, left-down
        for i in range(0, lx):
            f[3,i,j] = f[3,i+1,j]
            f[7,i,j] = f[7,i+1,j+1]

move()



