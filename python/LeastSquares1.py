get_ipython().magic('pylab inline')

from numpy.random import randn

# set the seed for the random number generator 
# so results are reproducible if we re-run notebook:
from numpy.random import seed
seed(135429)  

z0true = 350.
gtrue = 9.81

def z(t):
    return z0true - 0.5*gtrue*t**2

tdata = linspace(1,8,15)                # times for sample data
zdata = z(tdata) + 15*randn(len(tdata))  # sample data with random errors

plot(tdata, zdata, 'o')
axis([0,10, 0,400])
ylim
xlabel('time')
ylabel('height')

tfine = linspace(0,8,1000)
zfine = z(tfine)
plot(tfine, zfine, 'b')
title("Data points and true parabola")

A = vstack((ones(tdata.shape), -0.5*tdata**2)).T
print "A = \n", A

b = array([zdata]).T
print "b = \n", b

AA = dot(A.T, A)
print "A^*A = \n", AA
Ab = dot(A.T, b)
print "\nA^*b = \n", Ab

x = solve(AA, Ab)  
print "x = \n", x

z0fit = x[0]
gfit = x[1]
print "Estimate of g is %g, true value is %s"  % (gfit,gtrue)

zfit = z0fit - 0.5*gfit*tfine**2
plot(tfine, zfit, 'r')

plot(tdata, zdata, 'o')
axis([0,10, 0,400])
ylim
xlabel('time')
ylabel('height')
title("Sample data and parabola from least squares fit")

[U,S,Vstar] = svd(A)
Uhat = U[:,0:2]
print "Uhat =\n", Uhat

bhat = dot(Uhat, dot(Uhat.T,b))
print "bhat = Pb = \n", bhat

print "The rank of [A|b] is ", matrix_rank(hstack((A,b)))
print "The rank of [A|bhat] is ", matrix_rank(hstack((A,bhat)))

plot(tdata, zdata, 'bo', label="original b")
plot(tdata, bhat, 'ro', label="projected bhat")
plot(tfine, zfit, 'r-', label="best fit")
legend()  # makes a legend based on labels
axis([0,10, 0,400])
ylim
xlabel('time')
ylabel('height')
title("Sample data and parabola from least squares fit")



