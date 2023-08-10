get_ipython().magic('pylab inline')

from scipy import linalg as la

def KDparams(F):
    u, s, v = svd(F)
    Rxy = s[0]/s[1]
    Ryz = s[1]/s[2]
    K = (Rxy-1)/(Ryz-1)
    D = sqrt((Rxy-1)**2 + (Ryz-1)**2)
    return K, D

yearsec = 365.25*24*3600
sr = 3e-15

times = linspace(0.00000001,10,20)
alphas = linspace(0,90,20)
time, alpha = meshgrid(times, alphas)
K = zeros_like(alpha)
D = zeros_like(alpha)

for (r,c) in np.ndindex(alpha.shape):
    a = deg2rad(alpha[r,c])
    t = time[r,c]*1e6*yearsec
    edot = sr*sin(a)
    gdot = sr*cos(a)
    L = array([[0, gdot, 0], [0, -edot, 0],[0, 0, edot]])
    F = la.expm(L*t)
    K[r,c], D[r,c] = KDparams(F)
    

contourf(time, alpha, K, linspace(0, 1, 11))
colorbar()

contourf(time, alpha, D, linspace(0, 2.5, 11))
colorbar()

from IPython.core.display import HTML
def css_styling():
    styles = open("./css/sg2.css", "r").read()
    return HTML(styles)
css_styling()

