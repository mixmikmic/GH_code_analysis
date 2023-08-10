get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2  """Reloads all functions automatically"""')
get_ipython().magic('matplotlib notebook')

from irreversible_stressstrain import StressStrain as strainmodel
import test_suite as suite
import graph_suite as plot
import numpy as np

model = strainmodel('ref/HSRS/22').get_experimental_data()

slopes = suite.get_slopes(model)
second_deriv_slopes = suite.get_slopes(suite.combine_data(model[:-1,0],slopes))

# -- we think that yield occurs where the standard deviation is decreasing AND the slopes are mostly negative
def findYieldInterval(slopes, numberofsections):
    
    def numneg(val):
        return sum((val<0).astype(int))
    
    # -- divide into ten intervals and save stddev of each
    splitslopes = np.array_split(slopes,numberofsections)
    splitseconds = np.array_split(second_deriv_slopes,numberofsections)
        
    # -- displays the number of negative values in a range (USEFUL!!!)
    for section in splitslopes:
        print numneg(section), len(section)
        
    print "-------------------------------" 
    
    for section in splitseconds:
        print numneg(section), len(section)

    divs = [np.std(vals) for vals in splitslopes]
    
    # -- stddev of the whole thing
    stdev = np.std(slopes)
    
    interval = 0
    
    slopesect = splitslopes[interval]
    secondsect = splitseconds[interval]
    
    print divs, stdev
    
    # -- the proportion of slope values in an interval that must be negative to determine that material yields
    cutoff = 3./4.
    
    while numneg(slopesect)<len(slopesect)*cutoff and numneg(secondsect)<len(secondsect)*cutoff:
        
        interval = interval + 1
        
        """Guard against going out of bounds"""
        if interval==len(splitslopes): break
            
        slopesect = splitslopes[interval]
        secondsect = splitseconds[interval]                                       
    
    print 
    print interval
    return interval

numberofsections = 15
interval_length = len(model)/numberofsections

"""
Middle of selected interval

Guard against going out of bounds
"""
yield_interval = findYieldInterval(slopes,numberofsections)
yield_index = min(yield_interval*interval_length + interval_length/2,len(model[:])-1) 
yield_value = np.array(model[yield_index])[None,:]

print 
print yield_value

model = strainmodel('ref/HSRS/326').get_experimental_data()

strain = model[:,0]
stress = model[:,1]

slopes = suite.get_slopes(model)
second_deriv = suite.get_slopes(suite.combine_data(model[:-1,0],slopes))


"""Now what if we have strain vs slope"""
strainvslope = suite.combine_data(strain,slopes)
strainvsecond = suite.combine_data(strain,second_deriv)
plot.plot2D(strainvsecond,'Strain','Slope',marker="ro")
plot.plot2D(model,'Strain','Stress',marker="ro")

model = strainmodel('ref/HSRS/326').get_experimental_data()

strain = model[:,0]
stress = model[:,1]

slopes = suite.get_slopes(model)
second_deriv = suite.get_slopes(suite.combine_data(model[:-1,0],slopes))

num_intervals = 80

interval_length = len(second_deriv)/num_intervals
split_2nd_derivs = np.array_split(second_deriv,num_intervals)

print np.mean(second_deriv)
down_index = 0

for index, section in enumerate(split_2nd_derivs):
    if sum(section)<np.mean(slopes):
        down_index = index
        break
        
yield_index = down_index*interval_length

print strain[yield_index], stress[yield_index]

model = strainmodel('ref/HSRS/326').get_experimental_data()

strain = model[:,0]
stress = model[:,1]

first_deriv = suite.get_slopes(model)
second_deriv = suite.get_slopes(suite.combine_data(model[:-1,0],first_deriv))

plot1 = suite.combine_data(strain,first_deriv)
plot2 = suite.combine_data(strain,second_deriv)

plot.plot2D(model)
plot.plot2D(plot1)
plot.plot2D(plot2)

model = strainmodel('ref/HSRS/222').get_experimental_data()

strain = model[:,0]
stress = model[:,1]

first_deriv = suite.get_slopes(model)
second_deriv = suite.get_slopes(suite.combine_data(model[:-1,0],first_deriv))


ave_deviation = np.std(second_deriv)
deviation_second = [np.std(val) for val in np.array_split(second_deriv,30)]

yielding = 0


for index,value in enumerate(deviation_second):
    
    if value != 0.0 and value<ave_deviation and index!=0:
        yielding = index
        break
    
print second_deriv
#print "It seems to yield at index:", yielding
        
#print "These are all of the standard deviations, by section:", deviation_second, "\n"
#print "The overall standard deviation of the second derivative is:", ave_deviation

model = strainmodel('ref/HSRS/22').get_experimental_data()

strain = model[:,0]
stress = model[:,1]

first_deriv = suite.get_slopes(model)
second_deriv = suite.get_slopes(suite.combine_data(model[:-1,0],first_deriv))

print second_deriv;
return;

chunks = 20
int_length = len(model[:])/chunks

deriv2spl = np.array_split(second_deriv,chunks)
deviation_second = [abs(np.mean(val)) for val in deriv2spl]

del(deviation_second[0])
print deviation_second
print np.argmax(deviation_second)
#print "The standard deviation of all the second derivatives is", np.std(second_deriv)

import numpy as np

# -- climbs a discrete dataset to find local max
def hillclimber(data, guessindex = 0):
    
    x = data[:,0]
    y = data[:,1]
    
    curx = x[guessindex]
    cury = y[guessindex]
    
    guessleft = max(0,guessindex-1)
    guessright = min(len(x)-1,guessindex+1)
    
    done = False
    
    while not done:
        
        left  = y[guessleft]
        right = y[guessright]

        difleft = left-cury
        difright = right-cury

        if difleft<0 and difright<0 or (difleft==0 and difright==0):
            done = True
        elif difleft>difright:
            cur = left
            guessindex = guessleft
        elif difright>difleft or difright==difleft:
            cur = right
            guessindex = guessright
        
    return guessindex

func = lambda x: x**2
xs = np.linspace(0.,10.,5)
ys = func(xs)

data = suite.combine_data(xs,ys)
print hillclimber(data)
    

