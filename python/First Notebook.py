get_ipython().magic('matplotlib inline')

#adding necessary libraries
import matplotlib as mpl
import matplotlib.pyplot as plot
import numpy as np

plot.ylabel("Values of our Function!")
plot.plot([1,2,3,4],[1,5,1,5],'r')
plot.show()

t = np.arange(1.,50.)

def f(t):
    return 5*t+3

for sec in t:
    plot.plot(t,f(t))

#for sec in t:
#    plot.plot(t,(lambda t: 5*t+3))

t = np.arange(-50.,50.)

for sec in t:
    plot.plot(sec,1,"ro")
    
plot.show()

for x in t:
    plot.plot(x,x**2,"ro")

from scipy.optimize import minimize

for x in t:
    plot.plot(x,x**2,"ro")

function = lambda x: x**3

for x in t:
    plot.plot(x,function(x),'ro')

# Defining an arbitrary function that we will then minimize
def funct(x):
    return x**4-x**2+x-1

for x in t:    
    plot.plot(x,funct(x),'yo')

# our initial guess here will be bad, just to see how quickly it will work
minimum=minimize(funct,1000,method='Nelder-Mead',tol=1e-10)

print minimum.x

print funct(minimum.x)

# We can do the same thing with x^2 and see what we will have as a result
def quadratic(x):
    return x**2

quad_min=minimize(quadratic,10000000,method='Nelder-Mead',tol=1e-10)
print quad_min.x #0., we know this is correct because the analytic solution is zero

plot.plot(t,quadratic(t))
plot.title("Y=X^2")
plot.annotate("Minimum",xy=(quad_min.x,quadratic(quad_min.x)),xytext=(0,1000),arrowprops=dict(facecolor='black', shrink=0.05))

from sklearn import linear_model as lin_mod
linfit = lin_mod.LinearRegression()
#linfit.fit([0,0],[1,0],[1,1],[2,1])

# We create a sample data set, the first half will be used for learning, and the next half for testing
x = np.arange(0.,50.,0.5)
y = np.arange(0.,100.,1.)

# The actual model here is y=2x

teachx = x[:50]
testx = x[50:]

teachy = y[:50]
testy = y[50:]

#linfit.fit(teachx,teachy)

print teachx

print teachy

print "The first data set is of size", teachx.size, "The second is of size", teachy.size #confirming that we're good

teachx = teachx.reshape(-1,1);teachy = teachy.reshape(-1,1) #if reshape is -1, it infers the proper values for shape
linfit.fit(teachx,teachy)

print linfit.coef_ #2, correct, this is y=2*x

testx = testx.reshape(-1,1)
predicted = linfit.predict(testx).reshape(-1,1)

plot.plot(testx,predicted,"b-") # demonstrates the prediction was 100% accurate
plot.plot(testx,testy,"y-") #the two plots
plot.show() #comes out a mix of the two colors

import numpy as np
x = np.linspace(-50.,50.,101)
y = np.arange(-50.,51.,1.) #does not include last data point
print "X", x
print "Y", y

# arange() produces more accurate results than does linspace()

x = np.arange(-50.,51.,1.)
first_function = lambda val: val**3
print(first_function(3))

second_function = lambda val: val**3 + 6

print second_function(5)

first_domain = x[:50]
second_domain = x[50:]

#first_range = vectorize(first_function(first_domain)) #not how vectorize works, should be called on numpy
#print first_range

first_range = first_function(first_domain)
print first_range

second_range = second_function(second_domain)
print second_range

for num in first_domain:
    print "Num:",num, "Value:", first_function(num)

for num in second_domain:
    print "Num:",num, "Value:", second_function(num)
    
print second_function(second_domain)

from sklearn.naive_bayes import GaussianNB as naive

#first_data = first_function(first_domain)
#second_data = second_function(second_domain)

#print first_data.size, second_data.size

#merged_data = np.concatenate(first_data,second_data)
# we will now teach the model our data, and try to predict other points

first_data = first_function(first_domain)
second_data = second_function(second_domain)

merged_data = np.concatenate([first_data,second_data])

print merged_data # this merged_data contains some data from one model, and some from another

predict = naive()
prop_x = x.reshape(-1,1)
prop_data = merged_data.reshape(-1,1)
predicted_values = predict.fit(prop_x,prop_data).predict(prop_x)

print predicted_values-merged_data # either this fit was very good, or there is a problem, it displays no error here

# using n-degree polynomials
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

degree = 3 #we want to guess a cubic (because we already know it is in fact cubic)

model = make_pipeline(PolynomialFeatures(degree),Ridge())
model.fit(prop_x,prop_data)

x_plot = np.arange(-50.,50.,0.2).reshape(-1,1)

y_plot = model.predict(x_plot).reshape(-1,1)
plot.plot(x_plot,y_plot)

print model.predict(0) #perfect, essentially created a curve between the two 

tangent = np.tan
pi = np.pi

print tangent(0)
print tangent(np.pi/4) # our tangent works as numpy's tangent function now

# creating data points from tangent

boundary = 0.9

lower_bound = -pi/2 + boundary
upper_bound = pi/2 - boundary

num_datapoints = 1000
x_data = np.linspace(lower_bound,upper_bound,num_datapoints).reshape(-1,1)
y_data = tangent(x_data).reshape(-1,1)

magnitude_error = 1e16
best_approx = 0

# checking all degrees from 1 to 20
for degree in xrange(1,21):
    
    #print degree

    model = make_pipeline(PolynomialFeatures(degree),Ridge())
    model.fit(x_data,y_data)

    test_x = np.linspace(lower_bound,upper_bound,num_datapoints).reshape(-1,1)
    
    #plot.plot(test_x,model.predict(test_x))

    current_error = abs(np.linalg.norm(model.predict(test_x))-np.linalg.norm(y_data))
    
    if current_error < magnitude_error :
        best_approx = degree
        magnitude_error = current_error
        
    print current_error #the difference in the arrays in constant, the algorithm must normalize them in this way
    
    #plot.show()
    
print "The best approximation was of order",best_approx

import csv

ages = np.zeros(0)

print ages

with open("sample_data.csv") as file:
    readin = csv.reader(file)

    next(readin,None) #ignores the header
    
    for row in readin:
        
        # we also ignore the first column, because it's just a space
        for entry in xrange(1,len(row)):
            print row[entry]
            
        ages = np.append(ages,row[3])

sum = 0
for entry in ages:
    sum = sum+float(entry)
print "Average age is:", round(sum/len(ages))

# finding the minimum of x^2

from scipy.optimize import basinhopping as hop
func = lambda x: x**2+x**4+x**20-x**3+5+x**23

x0 = 500. #bad initial guess just to see how quickly the algorithm will converge to a result

# minimizer_kwargs used for internal minimization in algorithm (minimizer_kwargs={method:"BFGS" or "Nelder-Mead})
minimizer_kwargs = {"method":"Nelder-Mead"}
minimum = hop(func,x0,minimizer_kwargs=minimizer_kwargs,niter=1)
print "x =", minimum.x, "f(x) =", minimum.fun

# we can test this on the rosenbrock function to see how effective it is 
from scipy.optimize import rosen
minimum = hop(rosen,x0,minimizer_kwargs=minimizer_kwargs,niter=100)
print "x =", minimum.x[0], "f(x) =", minimum.fun

#import numpy. as sin 
#import numpy.cos as cos
#import numpy.sec as sec

#from numpy import sin, cos, sec

from numpy import sin, cos

sec = lambda x: 1/cos(x)

function = lambda x: cos(x)+sin(x)+3*sec(x)
x0 = 0 #initial guess
minimum = hop(function,x0,minimizer_kwargs=minimizer_kwargs,niter=5)
print "x =", minimum.x[0], "f(x) =", minimum.fun #ends up finding local minimum, abs minimum doesn't exist

#trying many guesses, certain guesses give us -inf, which is the global minimum, others gives us the local minimum
for guess in xrange(0,10):
    minimum = hop(function,guess,minimizer_kwargs=minimizer_kwargs,niter=5)
    print "x =", minimum.x[0], "f(x) =", minimum.fun

func2d = lambda x: x[0]**2+x[1]**2

minimizer_kwargs = {"method":"L-BFGS-B"}
x0 = [1.0, 1.0]
ret = hop(func2d, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
print("global minimum: x = [%.4f], f(x0) = %.4f" % (ret.x[0], ret.fun))

func2d = lambda x: x[0]**3+x[1]**2
#print func2d([1,2])
x0 = [1,5]
minimizer_kwargs = {"method":"L-BFGS-B",}
ret = hop(func2d, x0, minimizer_kwargs=minimizer_kwargs, niter=10000)
print("global minimum: x = [%.4f], f(x0) = %.4f" % (ret.x[0], ret.fun)) #prints arbitrarily small values, no G.L.B.

class first:
    pass

class Greeting:
    
    # not sure why the keyword "self" is needed in function definition
    def hello(self):
        return "Hello, World!"

hi = Greeting()
hi.hello()

# __init__ is a special function invoked when PeopleList() is called
class PeopleList:
    
    def __init__(self):
        self.people = []
        
    def add_person(self,name):
        
        #ensures that the user is giving a string as a parameter for the function
        if isinstance(name,str):
            self.people.append(name)
        
        else:
            return   
        
some_peeps = PeopleList()
some_peeps.add_person("John")
some_peeps.add_person("Tim")
some_peeps.add_person(3)
some_peeps.add_person("Suzy")

print some_peeps.people

#important to note that self must be a parameter in all class methods & global functions can be assigned to classes

# if we wanted to give PeopleList a new ability, such as returning a greeting
def whatsup(self):
    return "Hey!"

PeopleList.greeting = whatsup

print some_peeps.greeting()

#if the 'self' keyword does not serve a purpose then we can call it with a nonsense argument
whatsup(1)

class Chef:
    
    pastime = "Cooking"
    
    def __init__(self):
        pass
    
class Writer:
    
    hobby = "fishing"
    
class Archer:
    
    bow = "Ranger1000"
    
class Superhero(Chef,Writer,Archer):
    pass

# In this case superman inherited traits from all of the others
superman = Superhero()
print superman.pastime, superman.hobby, superman.bow

import scipy.optimize as optimize

hop = optimize.basinhopping
rosen = optimize.rosen
brute = optimize.brute

ranges = (slice(-500.,500.,1),slice(-500.,500.,1.))

output = brute(rosen,ranges,finish=optimize.fmin)
print output[0]
print output[1]

#takes a very long time.

guess = [20.,20.]

import timeit

start = timeit.default_timer()

result = hop(rosen,guess,niter=5)
print result.x[0], result.fun

stop = timeit.default_timer()

print "It took:",stop - start 



