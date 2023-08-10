import numpy
import scipy.signal
import scipy.optimize
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

G = scipy.signal.lti([2], [1, 2, 1])

plt.plot(*G.step())

M = 10  # Control horizon
P = 20  # Prediction horizon

DeltaT = 0.5  # Sampling rate

tcontinuous = numpy.linspace(0, DeltaT*P, 1000)  # some closely spaced time points
tpredict = numpy.arange(0, P*DeltaT, DeltaT)   # discrete points at prediction horizon

u = numpy.ones(M)
r = numpy.ones(P)

tau_c = 0.5
r = 1 - numpy.exp(-tpredict/tau_c)

x0 = numpy.zeros(G.A.shape[0])

def extend(u):
    return numpy.concatenate([u, numpy.repeat(u[-1], P-M)])

def prediction(u, t=tpredict, x0=x0):
    t, y, x = scipy.signal.lsim(G, u, t, X0=x0, interp=False)
    return y

plt.plot(prediction(extend(u)))

def objective(u, x0=x0):
    y = prediction(extend(u))
    umag = numpy.abs(u)
    constraintpenalty = sum(umag[umag > 2])
    movepenalty = sum(numpy.abs(numpy.diff(u)))
    strongfinish = numpy.abs(y[-1] - r[-1])
    return sum((r - y)**2) + 0*constraintpenalty + 10*movepenalty + 100*strongfinish

objective(u)

result = scipy.optimize.minimize(objective, u)
uopt = result.x
result.fun

ucont = extend(uopt)[((tcontinuous-0.01)//DeltaT).astype(int)]

plt.figure()
plt.plot(tcontinuous, ucont)
plt.xlim([0, DeltaT*(P+1)])
plt.figure()
plt.plot(tcontinuous, prediction(ucont, tcontinuous), 
         tpredict, prediction(extend(uopt)),
         tpredict, r,
         )





