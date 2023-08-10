from scipy import optimize

get_ipython().magic('pylab inline')

def drawrosen(ax):
    x = linspace(-2, 2, 200)
    y = linspace(-2, 2, 200)
    xx, yy = numpy.meshgrid(x, y)
    map = optimize.rosen(array([xx.ravel(), yy.ravel()]).reshape(2, -1)).reshape(xx.shape)
    ax.contour(x, y, log10(map + 1e-9), vmin=-4, cmap=cm.coolwarm) #levels=(1, 2, 4, 8, 16, 32))
    

def walk(axes, method, *args, **kwargs):
    x0 = (-1.4, -1.325)
    l = [x0]
    i = [x0]
    def myfunc(x):
        l.append(x.copy())
        return optimize.rosen(x)
    def cb(x):
        i.append(x.copy())
    r = optimize.minimize(myfunc, x0, jac=optimize.rosen_der,
                            hess=optimize.rosen_hess, #hessp=optimize.rosen_hess_prod,
                          callback=cb,
                          method=method, options=dict(disp=False))
    print(method, r)
    l.append(r.x.copy())
    x = array(l).T[0]
    y = array(l).T[1]
    axes[0].plot(x, y, 'x', **kwargs)
    i.append(r.x.copy())
    x = array(i).T[0]
    y = array(i).T[1]
    axes[0].plot(x, y, 'o-', markerfacecolor='none', **kwargs)
    axes[0].set_xticks(linspace(-1.5, 1.5, 7, endpoint=True))
    axes[0].set_yticks(linspace(-1.5, 1.5, 7, endpoint=True))
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[1].plot(optimize.rosen(array(l).T))
    axes[1].set_yscale('log')
    axes[1].set_ylim(3e-4, 5e3)
    axes[1].set_xlim(0, 50)
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.set_ticks_position('right')
    axes[1].text(0.93, 0.8, method, ha='right', va='top', transform=axes[1].transAxes)
    if not hasattr(r, 'njev'): r.njev=0
    if not hasattr(r, 'nhev'): r.nhev=0
    axes[1].text(0.93, 0.7, 'nfev=%d\nnjev=%d\nnhev=%d' % (r.nfev, r.njev, r.nhev),
                 ha='right', va='top', transform=axes[1].transAxes)

gsleft = GridSpec(2, 2)
gsright = GridSpec(2, 2)
gsleft.update(left=0.05, right=0.45, wspace=0.05, hspace=0)
gsright.update(left=0.58, right=0.98, wspace=0.05, hspace=0)

fig = figure(figsize=(8, 6))
def makepair(gs, i):
    ax1 = fig.add_subplot(gs[i, 0])
    ax2 = fig.add_subplot(gs[i, 1])
    if i == 0:
        ax1.set_xticks([])
        ax2.set_xticks([])
    return [ax1, ax2]

axes = makepair(gsleft, 0)
drawrosen(axes[0])
walk(axes, "Nelder-Mead", '+-', color='k')
axes = makepair(gsleft, 1)
drawrosen(axes[0])
walk(axes, "CG", '+-', color='r')
axes = makepair(gsright, 0)
drawrosen(axes[0])
walk(axes, "BFGS", 'x-', color='g')
axes = makepair(gsright, 1)
drawrosen(axes[0])
#walk(axes[3], "Newton-CG", '+-', color='b')
walk(axes, "trust-ncg", '+-', color='b')

savefig('Rosenbrock.pdf', dpi=200)
#trust-ncg 





