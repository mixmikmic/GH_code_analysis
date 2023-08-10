#
# Simple python program to calculate s as a function of t. 
# Any line that begins with a '#' is a comment.
# Anything in a line after the '#' is a comment.
#

lam=0.01            # define some variables: lam, dt, s, s0 and t. Set initial values.
dt=1.0
s=s0=100.0
t=0.0

def f_s(s,t):         # define a function that describes the rate of change of 's'
    return -lam*s

print ('t     s')     # print a header for the table of data
for i in range(11):
    print (t,s)       # iterate through 11 steps, starting at 0
    ds = f_s(s,t)*dt  # compute the change in 's' using the 'rule' that ds/dt = f(s,t)
    s = s + ds      # update s
    t = t + dt      # update t

get_ipython().magic('pylab inline')

slist=[]
tlist=[]

lam=0.01
dt=1.0
s=s0=100.0
t=0.0

tlist.append(t)
slist.append(s)

print ('t     s')
for i in range(11):
    s += f_s(s,t)*dt
    t += dt
    tlist.append(t)
    slist.append(s)

#plot(tlist, slist, 'b.', tlist, 100.0*exp(-lam*array(tlist)))
print ("tlist=", tlist)
print ("slist=", slist)

exact = s0*exp(-lam*array(tlist))
print ("exact", exact)

title('Decay Results')
xlabel('time (s)')
ylabel('n (nuclei)')
plot(tlist, slist, 'b.', tlist, exact, 'r-')

#
# Here is the raw data for the position of the muffin cup as a function of time. Use the "split" function to break it into
# a list of (possibly empty) strings.
#

data = """0.000000000E0	-2.688162330E0
3.336670003E-2	-4.301059729E0
6.673340007E-2	-5.376324661E0
1.001001001E-1	-6.989222059E0
1.334668001E-1	-1.129028179E1
1.668335002E-1	-1.451607658E1
2.002002002E-1	-2.043003371E1
2.335669002E-1	-2.526872591E1
2.669336003E-1	-3.118268303E1
3.003003003E-1	-3.870953756E1
3.336670003E-1	-4.623639208E1
3.670337004E-1	-5.430087907E1
4.004004004E-1	-6.236536606E1
4.337671004E-1	-7.150511799E1
4.671338005E-1	-8.010723744E1
5.005005005E-1	-8.924698937E1
5.338672005E-1	-9.892437376E1
5.672339006E-1	-1.080641257E2
6.006006006E-1	-1.177415101E2
6.339673006E-1	-1.274188945E2
6.673340007E-1	-1.370962788E2
7.007007007E-1	-1.467736632E2
7.340674007E-1	-1.575263126E2
7.674341008E-1	-1.672036969E2
8.008008008E-1	-1.768810813E2
8.341675008E-1	-1.865584657E2
8.675342009E-1	-1.973111150E2
9.009009009E-1	-2.075261319E2
9.342676009E-1	-2.182787812E2
9.676343010E-1	-2.284937981E2
""".splitlines()  # split this string on the "newline" character.

print("We have", len(data), "data points.")

#
# Here we'll take the list of strings defined above and break it into actual numbers in reasonable units.
#

tlist = []
ylist = []
for s in data:
    t,y = s.split()     # break string in two
    t=float(t)          # convert time to float
    y=float(y)/100.0    # convert distanct (in meters) to float
    tlist.append(t)
    ylist.append(y)
        
print ("tlist=",tlist)
print ("ylist=",ylist)

plot(tlist, ylist)

vlist = []  # Velocity list (computed velocities from experimental data)
tvlist = []  # time list (times for corresponding velocities)
for i in range(1,len(tlist)):
    dy=ylist[i]-ylist[i-1]
    dt=tlist[i]-tlist[i-1]
    vlist.append(dy/dt)
    tvlist.append((tlist[i]+tlist[i-1])/2.0)
    
plot(tvlist,vlist,'g.')

m=0.0035  # kg
g=9.8     # m/s
b=0.001    # total guess, need to improve

v=0.0     # start with zero velocity

dt = (tlist[-1]-tlist[0])/(len(tlist)-1)  # time per frame in original video
t=0.0

vclist = [v]
tclist = [t]

def deriv(v, t):
    return b*v**2/m - g

for i in range(len(tlist)):
    dv = deriv(v,t)*dt
    v += dv
    t += dt
    
    vclist.append(v)
    tclist.append(t)
    
title("Comparison of experimental and drag model")
xlabel("time(s)")
ylabel("velocity (m/s)")
plot(tclist, vclist, 'r-',tvlist,vlist,'g.')



