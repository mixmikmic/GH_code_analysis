get_ipython().magic('matplotlib inline')
get_ipython().magic('pdb on')
get_ipython().magic('run _547')

np.set_printoptions(precision=2)

g,m,I = 9.81,1.,1. # m/sec^2, kg, kg m^2

def f(t,x,u):
    q,dq = x[:3],x[3:] # positions, velocities
    h,v,theta = q # horiz., vert., rotation
    u1,u2 = u # thrust, torque
    return np.hstack([dq,(u1/m)*np.sin(theta),
                        -g + (u1/m)*np.cos(theta),
                        u2/I])

def h(t,x,u):
    q,dq = x[:3],x[3:] # positions, velocities
    h,v,theta = q # horiz., vert., rotation
    return np.array([h,v]) # horizontal, vertical position

q0 = np.array([0.,1.,0.])
dq0 = np.array([0.,0.,0.])
x0 = np.hstack((q0,dq0))
u0 = np.array([m*g,0.])
print 'x0 =',x0,'\nf(x0) =',f(0.,x0,u0)

dt = 1e-2 # coarse timestep
freq = .5 # one cycle every two seconds
t = 2./freq # two periods
q = [0.,.1,0.] # start 10cm up off the ground
dq = [0.,0.,0.] # start with zero velocity
x = np.hstack((q,dq))

# input is a periodic function of time
ut = lambda t : np.array([m*g + np.sin(2*np.pi*t*freq),0.])
# lambda is a shorthand way to define a function
# -- equivalently:
def u(t):
    return np.array([m*g + np.sin(2*np.pi*t*freq),0.])

sim = forward_euler
t_,x_ = sim(f,t,x,dt=dt,ut=ut)
u_ = np.array([u(t) for t in t_])
# sim() returns arrays t_ and x_
# x_[j] is the state of the system (i.e. pos. and vel.) at time t_[j]

fig = plt.figure(figsize=(4,8));

ax = plt.subplot(311)
ax.plot(t_,x_[:,:3],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('position')
ax.legend([r'$h$',r'$v$',r'$\theta$'],ncol=3,loc='right')

ax = plt.subplot(312)
ax.plot(t_,x_[:,3:],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('velocity')

ax = plt.subplot(313)
ax.plot(t_,u_,'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('input')

get_ipython().magic('run _anim')

fig, ax = plt.subplots(figsize=(4,4)); ax.axis('equal'); ax.grid('on');

line, = ax.plot([], [], 'b', lw=2);

# initialization -- called once
def init():
    gndh,gndv = [-10.,10.,10.,-10.],[0.,0.,-5.,-.5]
    ax.fill(gndh,gndv,'gray')
    line.set_data([], [])
    ax.set_xlim(( -1., 1.))
    ax.set_ylim(( -.15, 2.))
    return (line,)

# animation -- called iteratively
def animate(t):
    j = (t_ >= t).nonzero()[0][0]
    h,v,th = x_[j,:3]
    w = .25
    x = np.array([-w/2.,w/2.,np.nan,0.,0.])
    y = np.array([0.,0.,np.nan,0.,+w/3.])
    z = (x + 1.j*y)*np.exp(1.j*th) + (h + 1.j*v)
    line.set_data(z.real, z.imag)
    return (line,)

plt.close(fig)

# call the animator
animation.FuncAnimation(fig, animate, init_func=init, repeat=True,
                        frames=np.arange(0.,t_[-1],.1), interval=20, blit=True)

A = D(lambda x : f(0.,x,u(0)),x)
B = D(lambda u : f(0.,x,u),u(0))

print 'A =\n',A,'\n','B =\n',B

from control import lyap

W = lyap(-1.5*np.identity(6)-A,np.dot(B,B.T))
K = .5*np.dot(B.T,la.inv(W))

print "closed loop stable?",np.all(np.array(la.eigvals(A - np.dot(B,K))).real < 0)

ux = lambda x : np.dot(x - x0, -K.T) + u0

np.random.seed(50)

dt = 1e-2
t = 3. 
x = x0 + 3*(np.random.rand(6)-.5)

# input is now a function of state
ux = lambda x : np.dot(x - x0, -K.T) + u0

t_,x_ = sim(f,t,x,dt=dt,ux=ux)
u_ = np.array([ux(x) for x in x_])

fig = plt.figure(figsize=(4,8));

ax = plt.subplot(311)
ax.plot(t_,x_[:,:3],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('position')
ax.legend([r'$h$',r'$v$',r'$\theta$'],ncol=3,loc='right')

ax = plt.subplot(312)
ax.plot(t_,x_[:,3:],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('velocity')

ax = plt.subplot(313)
ax.plot(t_,u_,'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('input')

fig, ax = plt.subplots(figsize=(4,4)); ax.axis('equal'); ax.grid('on');

line, = ax.plot([], [], 'b', lw=2);

plt.close(fig)

# call the animator
animation.FuncAnimation(fig, animate, init_func=init, repeat=True,
                        frames=np.arange(0.,t_[-1],.1), interval=20, blit=True)



