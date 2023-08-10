get_ipython().magic('pylab inline')

m=1.0    # assume one kilogram
F0=1.0   # and one newton, just for illustration

v0 = 0.0  # start at rest
x0 = 0.0  # at the origin
xf = 3.0  # go to 3.0m
dt = 0.1  # 0.1 sec intervals
t = 0.0   # start at t=0.0s

s=array([x0, v0])  # the "state" will be position and velocity

def derivs_F(s, t):

    x=s[0]     # extract position and velocity from the "state"
    v=s[1]
    
    dxdt=v     # use the recipe here to get dvdt 
    dvdt=F0/m  # and dxdt
    
    return array([dxdt, dvdt])

def HeunStep(s, t, derivs, dt):
    f1=derivs(s,t)
    f2=derivs(s+f1*dt,t+dt)
    return s + 0.5*(f1+f2)*dt

xlist = [x0]
tlist = [t]

while s[0] < xf:
    s = HeunStep(s, t, derivs_F, dt)
    t += dt
    xlist.append(s[0])
    tlist.append(t)

title('Motion with constant force')
xlabel('time (s)')
ylabel('position (m)')
plot(tlist, xlist, 'r.')



