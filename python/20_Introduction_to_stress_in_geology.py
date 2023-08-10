get_ipython().magic('pylab inline')

S = array([[10, 0], [0, 5]])
tau = []
sn = []

for i in range(5000):
    theta = pi*uniform()
    n = array([cos(theta), sin(theta)])
    sv = dot(S, n)
    sn.append(dot(sv, n))
    tau.append(norm(sv-n*dot(sv, n)))

plot(sn, tau, 'k.')
axis('equal')
margins(x=0.1, y=0.1)
show()

def rand_vec():
    v = normal(size=3)
    return v / norm(v)

S = array([[5, 0, 0], [0, 9, 0], [0, 0, 15]])
tau = []
sn = []

for i in range(5000):
    n = rand_vec()
    sv = dot(S, n)
    sn.append(dot(sv, n))
    tau.append(norm(sv-n*dot(sv, n)))

plot(sn, tau, 'k.')
axis('equal')
margins(x=0.1, y=0.1)
show()

from IPython.core.display import HTML
def css_styling():
    styles = open("./css/sg2.css", "r").read()
    return HTML(styles)
css_styling()

