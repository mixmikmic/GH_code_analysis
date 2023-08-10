get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

S = numpy.linspace(0., 1., num=1000)
S = S[:, numpy.newaxis]
rww = models.ReducedWongWang()

rww

Io = numpy.linspace(0.00, 0.42, num=100)

dS = numpy.zeros((100, 1000))
fig = plt.figure(figsize=(12, 10))
for i, io in enumerate(Io):
    rww.I_o = io 
    dS[i, :]= rww.dfun(S[:, numpy.newaxis].T, numpy.array([[0.0]]))
    plt.plot(S, dS[i, :].T, 'k', alpha=0.1)
print S.shape

plt.plot(S, dS[0, :].T, 'k', lw=4, alpha=0.7, label =r"I_o=%0.2f" % Io[0])
plt.plot(S, dS[63, :].T, 'b',lw=4, alpha=0.7, label =r"I_o=%0.2f" % Io[63])
plt.plot(S, dS[77, :].T, 'r',lw=4, alpha=0.7, label =r"I_o=%0.2f" % Io[77])
plt.plot(S, dS[99, :].T, 'g',lw=4, alpha=0.7, label =r"I_o=%0.2f" % Io[99])
plt.plot(S, numpy.zeros(S.shape), 'k--')
plt.ylim([-0.01, 0.01])
plt.xlabel('S')
plt.ylabel('dS')
#plt.legend()

dS = numpy.zeros((100, 1000))
fig = plt.figure(figsize=(12, 10))
rww.w = 1.0
for i, io in enumerate(Io):
    rww.I_o = io 
    dS[i, :]= rww.dfun(S[:, numpy.newaxis].T, numpy.array([[0.0]]))
    plt.plot(S, dS[i, :].T, 'k', alpha=0.1)
print S.shape

plt.plot(S, dS[0, :].T, 'k', lw=4, alpha=0.7, label =r"I_o=%0.4f" % Io[0])
plt.plot(S, dS[70, :].T, 'b',lw=4, alpha=0.7, label =r"I_o=%0.4f" % Io[70])
plt.plot(S, dS[75, :].T, 'b',lw=4, alpha=0.5, label =r"I_o=%0.4f" % Io[75])
plt.plot(S, dS[76, :].T, 'r',lw=4, alpha=0.5, label =r"I_o=%0.4f" % Io[76])
plt.plot(S, dS[80, :].T, 'r',lw=4, alpha=0.7, label =r"I_o=%0.4f" % Io[80])
plt.plot(S, dS[99, :].T, 'g',lw=4, alpha=0.7, label =r"I_o=%0.4f" % Io[99])
plt.plot(S, numpy.zeros(S.shape), 'k--')
plt.ylim([-0.01, 0.01])
plt.xlabel('S')
plt.ylabel('dS')
#plt.legend()

W = numpy.linspace(0.8, 1.05, num=100)

rww.I_o=0.325

dS = numpy.zeros((100, 1000))
fig = plt.figure(figsize=(12, 10))
rww.w = 1.0
for i, w in enumerate(W):
    rww.w = w 
    dS[i, :]= rww.dfun(S[:, numpy.newaxis].T, numpy.array([[0.0]]))
    plt.plot(S, dS[i, :].T, 'k', alpha=0.1)
print S.shape

plt.plot(S, dS[0, :].T, 'k', lw=4, alpha=0.7, label =r"w=%0.4f" % W[0])
plt.plot(S, dS[60, :].T, 'b',lw=4,  alpha=0.7, label =r"w=%0.4f" % W[60])
plt.plot(S, dS[62, :].T, 'b',lw=4, alpha=0.3, label =r"w=%0.4f" % W[62])
plt.plot(S, dS[79, :].T, 'r',lw=4, alpha=0.3, label =r"w=%0.4f" % W[79])
plt.plot(S, dS[81, :].T, 'r',lw=4, alpha=0.7, label =r"w=%0.4f" % W[81])
plt.plot(S, dS[99, :].T, 'g',lw=4, alpha=0.7, label =r"w=%0.4f" % W[99])
plt.plot(S, numpy.zeros(S.shape), 'k--')
plt.ylim([-0.0001, 0.0001])
plt.xlabel('S')
plt.ylabel('dS')
#plt.legend()

