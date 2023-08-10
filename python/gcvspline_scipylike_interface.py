get_ipython().magic('pylab inline')

from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from gcvspline import GCVSmoothedNSpline, MSESmoothedNSpline, DOFSmoothedNSpline, SmoothedNSpline
#from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline

x = np.linspace(-3, 3, 50)
y0 = np.exp(-x**2)
np.random.seed(1234)

n = 0.1 * np.random.normal(size=50)
w = 1.0 / (np.ones_like(n) * std(n))
y = y0 + n

xs = np.linspace(-3, 3, 1000)

DX_auto = UnivariateSpline(x, y, w=w)
GCV_auto = GCVSmoothedNSpline(x, y, w=w)

ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
ax.plot(xs, GCV_auto(xs), label='GCV')
ax.plot(xs, DX_auto(xs), label='DX')
legend()

GCV_mse = MSESmoothedNSpline(x, y, w=w, variance_metric=GCV_auto.variance_metric)
GCV_dof = DOFSmoothedNSpline(x, y, w=w, dof=GCV_auto.dof)
GCV_manual = SmoothedNSpline(x, y, w=w, p=GCV_auto.p)

print(GCV_mse.variance_metric, GCV_mse.mse, GCV_mse.msr, GCV_mse.dof, GCV_mse.p)
print(GCV_dof.variance_metric, GCV_dof.mse, GCV_dof.msr, GCV_dof.dof, GCV_dof.p)
print(GCV_auto.variance_metric, GCV_auto.mse, GCV_auto.msr, GCV_auto.dof, GCV_auto.p)
print(GCV_manual.variance_metric, GCV_manual.mse, GCV_auto.msr, GCV_manual.dof, GCV_manual.p)

plot(xs, GCV_auto(xs) - GCV_mse(xs), label='GCV - MSE')
plot(xs, GCV_auto(xs) - GCV_dof(xs), label='GCV - DOF')
plot(xs, GCV_auto(xs) - GCV_manual(xs), label='GCV - Manual')
plot(xs, GCV_auto(xs) - DX_auto(xs), label='GCV - DX')
ylim(-std(n), std(n))
legend()

def check_w2(ax):
    r = []
    for wf in [0.1, 1.0, 10.0]:
        GCV2 = GCVSmoothedNSpline(x, y, w=w * wf)
        ax.plot(xs, GCV2(xs), label='w= w * %g' % (wf))
        r.append([wf, GCV2.mse, GCV2.dof])
        print('variance of the first data item is', GCV2.variance[0][0])
    return np.array(r)
ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
r = check_w2(ax)

def check_p(ax):
    r = []
    for pf in [0.1, 1.0, 10.0]:
        p = GCV_auto.p * pf
        GCV_manual = SmoothedNSpline(x, y, w=w, p=p)
        ax.plot(xs, GCV_manual(xs), label='p=%g' % p)
        r.append([p, GCV_manual.mse, GCV_manual.dof])
    return r

ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_p(ax)
legend()

def check_p2(ax):
    r = []
    for wf in [0.1, 1.0, 10.0]:
        GCV_manual = SmoothedNSpline(x, y, w=w * wf, p=GCV_auto.p * wf ** 2)
        ax.plot(xs, GCV_manual(xs), label='w= w * %g, p=p * %g' % (wf, wf**2))
        r.append([wf, GCV_manual.mse, GCV_manual.dof])
    return r

ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_p2(ax)
legend()

def check_mse(ax):
    for variance_metric in [0.1, 1.0, 10.0]:
        GCV_mse = MSESmoothedNSpline(x, y, w=w, variance_metric=variance_metric)
        ax.plot(xs, GCV_mse(xs), label='variance metric=%g' % variance_metric)
        #print(GCV_mse.mse, GCV_mse.dof, variance_metric)

ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_mse(ax)
legend()

def check_mse2(ax):
    for variance_metric in [0.1, 1.0, 10.0]:
        GCV_mse = MSESmoothedNSpline(x, y, w=w * variance_metric ** 0.5, 
                                     variance_metric=variance_metric)
        ax.plot(xs, GCV_mse(xs), label='variance metric=%g' % variance_metric)
        #print(GCV_mse.mse, GCV_mse.dof, variance_metric)
ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_mse2(ax)
legend()        

def check_dof(ax):
    for doff in [0.5, 1.0, 2.0]:
        # not a particular model, but looked like a good heuristic
        dof = len(y) - len(y) ** 0.5 * doff
        GCV_dof = DOFSmoothedNSpline(x, y, dof=dof, w=w)
        ax.plot(xs, GCV_dof(xs), label='dof=%g' % dof)
        print(GCV_dof.dof, dof)

ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_dof(ax)
legend()

def check_dof2(ax):
    for wf in [0.1, 1.0, 10.0]:
        # not a particular model, but looked like a good heuristic
        GCV_dof = DOFSmoothedNSpline(x, y, dof=GCV_auto.dof, w=w * wf)
        ax.plot(xs, GCV_dof(xs), label='wf=%g' % wf)
        #print(GCV_dof.dof, dof)
ax = subplot(111)
ax.plot(x, y, 'ro', ms=5)
check_dof(ax)
legend()

