for i in range(10):
    print i+1

for i in range(1,11):
    print i

N=3
for i in range(N):
    print i+1

def do_print(s, N):
    for i in range(N):
        print('%s, %i' % (s, i))
    
do_print('Hallo', 3)
do_print('Welt', 5)

for x in range(10): print x+1  # some alternative way of looping

for x in range(100):
    if x == 55:
        print('****** %i ******' % x)
    else:
        print(x)



def sboltz(T, e):
    return e*5.67E-8*T**4.  # W/m**2

def wien(T):
    return 2897.8/T  #returns lmax in mym

def radiation(T, e=1.):
    Q = sboltz(T, e)
    l_max = wien(T)
    return Q, l_max

print radiation(6000.)

print radiation(300., e=0.9)


import math
def wvpressure(T, rh):
    # uses the empirical Magnus formula for air over open water bodies
    # validity limited to certain temperature regions
    
    # perform some validity checks
    assert rh >= 0.
    assert rh <= 1.
    assert T > -45., 'Invalid temperature!'
    assert T < 60.
    
    es = 6.112 * math.exp((17.62*T)/(243.12+T)) * 100.    # factor 100 as results should be in Pa
    return es*rh, es
    
T=50.
rh = 1.
    
e, es = wvpressure(T, rh)

print(T, e, es)

# search for dewpoint (the quick and dirty way ...)
T = 10.
rh = 0.8

e, es = wvpressure(T, rh)

# now we have the actual water vapor pressure and need to search for the temperature where this corresponds to Es
# we use an itterative solution here

t0 = -40.
delta = 999999999.
dt = 0.5
t = t0*1.
tsol = t0*1.
tmax = T
while t < tmax:  # this is not good practice to do it, but we will learn only later how to do it better ...
    E, ES = wvpressure(t, rh)
    if abs(ES-e) < delta:
        delta = abs(ES-e)
        tsol = t*1.
    t += dt
print('The dew point was found at a temperature of %f degrees Celsius' % tsol)


