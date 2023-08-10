get_ipython().magic('pylab inline')

PS = array([[2, 0], [0, 0.5]])
SS = array([[1, 1], [0, 1]])

# simple shear first
dot(PS, SS)

# pure shear first
dot(SS, PS)

PSi = array([[1.01, 0], [0, 1/1.01]])
SSi = array([[1, 0.01], [0, 1]])

matrix_power(PSi, 10)

1.01**10

matrix_power(SSi, 10)

10*1.01

# function to calculate axial ratio of strain ellipse
def axialratio(F):
    u,s,v = svd(F)
    return s[0]/s[1]
# function to calculate orientation of long axis of the strain ellipse (counterclockwise from horizontal)
def orientation(F):
    u,s,v = svd(F)
    return rad2deg(arctan2(u[1,0], u[0,0]))

F = array([[1, 1], [0, 1]])

axialratio(F)

orientation(F)

nrange = arange(1, 100)

PSar = [axialratio(matrix_power(PSi, n)) for n in nrange]
SSar = [axialratio(matrix_power(SSi, n)) for n in nrange]

plot(nrange, PSar, label='pure shear')
plot(nrange, SSar, label='simple shear')
plt.legend(loc=2)

nrange = arange(1, 1000)
ang = [orientation(matrix_power(SSi, n)) for n in nrange]

plot(nrange, ang)

from IPython.core.display import HTML
def css_styling():
    styles = open("./css/sg2.css", "r").read()
    return HTML(styles)
css_styling()



