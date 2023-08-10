import graph_suite as plot
import parser
reload(parser)
import material_analytics
reload(material_analytics)
import numpy as np

#add to GUI
elastic, plastic = material_analytics.kmeanssplit(parser.stress_strain('ref/1150.dat').get_experimental_data()[:,:2])
plot.plotmult2D(elastic,plastic, title = 'Stress vs Strain', xtitle = "Strain ($\epsilon$)", ytitle= "Stress ($\sigma$)")

def yield_classic(data, cutoff = 0.1):
    
    startpt = cutoff # can't be zero or log will freak out
    
    data = material_analytics.log_prep(data,cutoff = cutoff)
    bestfit = material_analytics.log_approx(data)
    domain = np.linspace(startpt,max(data[:,0]),100001)

    vals = material_analytics.combine_data(domain,bestfit(domain))

    deriv1st = material_analytics.get_slopes(material_analytics.combine_data(domain,bestfit(domain)))

    find_nearest = lambda array,value: array[(np.abs(array-value)).argmin()]

    ave_slope = (vals[-1,1]-vals[0,1])/(vals[-1,0]-vals[0,0])
    index_ave = np.where(deriv1st == find_nearest(deriv1st, ave_slope))
    log_bend = vals[index_ave][0]
    young_mod = (log_bend[1]-vals[0,1])/(log_bend[0]-vals[0,0])

    linear_mod_2 = lambda x: young_mod*(x-0.2) + vals[0,1] # we will find where this intersects our exp data
    samples2 = material_analytics.samplepoints(linear_mod_2,[startpt,max(vals[:,0])],1001)

    diff = np.abs(samples2[:,1] - vals[:,1])

    """We can call this the intersection, its the closest point"""
    leastindex = diff.argmin()
    cross_est = vals[leastindex].reshape(1,-1)[0,1]
    yieldstress = find_nearest(data[:,1].reshape(-1,1),cross_est)[0]

    yieldpt = data[np.where(data[:,1] == yieldstress)]
    return yieldpt[0]

data = parser.stress_strain('ref/HSRS/326').get_experimental_data()
yieldpt = yield_classic(data)
index = np.where(data == yieldpt)[0][0]

elastic, plastic = data[:index], data[index:]
plot.plotmult2D(elastic,plastic, title = 'Stress vs Strain', xtitle = "Strain ($\epsilon$)", ytitle= "Stress ($\sigma$)")

data = parser.stress_strain('ref/900.dat').get_experimental_data()
yieldpt = material_analytics.yield_stress(data)[0]

index = np.where(data == yieldpt)[0][0]
elastic, plastic = data[:index+1], data[index:]

model = material_analytics.stress_model(data)

plot.plotmult2D(elastic,plastic, marker2 = 'r-', title = 'Stress vs Strain', xtitle = "Strain ($\epsilon$)", ytitle= "Stress ($\sigma$)")

data = parser.stress_strain('ref/1150.dat').get_experimental_data()

yldpt = material_analytics.yield_stress(data, cutoff=0.0000)
plot.plotmult2D(data, yldpt)

data = parser.stress_strain('ref/1150.dat').get_experimental_data()

find_nearest = lambda array,value: array[(np.abs(array-value)).argmin()]
#logft = material_analytics.log_approx(data)
yld = yield_classic(data)

def sort_by_x(array):
    
    x_values = np.copy(array[:,0])
    minimum_indices = []
    
    while(x_values.size != 0):
    
        minimum = np.inf
        minimum_index = np.nan
    
        """We find a minimum value in the array, and each time we do, we take it to the front"""
        for index, num in enumerate(x_values):

            if num < minimum:
                minimum = num
                minimum_index = index
                
        """For each iteration, we remove the index that was the minimum and add it to the list of minimum indices"""
        minimum_indices = np.append(minimum_indices, minimum_index)
        x_values = np.delete(x_values, minimum_index)
        
    to_return = np.zeros(0)
    
    for counter, index in enumerate(minimum_indices):
        print array[index][None,]
        to_return = np.vstack((to_return,array[index][:,None]))
            
    return to_return

example_data = material_analytics.combine_data([1,3,2,1],[1,4,5,6])
sort_by_x(example_data)











































import parser
reload(parser)

import material_analytics
reload(material_analytics)

import graph_suite as plot
import numpy as np

data = parser.stress_strain('ref/1150.dat').get_experimental_data()

av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0]) #rise over run
closest_index = lambda data, value: (np.abs(data-value)).argmin()

# no estimation needed for accurate data
deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

bend = closest_index(deriv1st[:,1],av_slope)
    
print "The average slope is", av_slope
print "The index where the curve bends is", bend

plot.plotmult2D(data,data[bend][None,:])

logapprox = material_analytics.log_approx(data)

data_x = np.linspace(min(data[:,0]),max(data[:,0]),1001)
data_y = logapprox(data_x)
data = material_analytics.combine_data(data_x,data_y)

av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0]) #rise over run
closest_index = lambda data, value: (np.abs(data-value)).argmin()

# no estimation needed for accurate data
deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

bend = closest_index(deriv1st[:,1],av_slope)
    
print "The average slope is", av_slope
print "The index where the curve bends is", bend

plot.plotmult2D(data,data[bend][None,:])

young_modulus = (data[bend,1]-data[0,1])/(data[bend,0]-data[0,0])

# this is the offset line
def linear_estimation(x):
    return data[0,1] + young_modulus*(x-0.002)

linear_y = linear_estimation(data_x)
difference_bw_est = np.abs(data_y-linear_y)
intersection = data[np.where(difference_bw_est[:]==min(difference_bw_est))[0]]

# want to find the intersection between the estimated line and data so find the closest pair of points
plot.plotmult2D(data,intersection)















offset = 0.002
cutoff = 0.00001

"""Get data"""
data_original = parser.stress_strain('ref/850.dat').get_experimental_data()

"""Fit a log curve"""
data = material_analytics.log_prep(data_original, cutoff = cutoff)
logapprox = material_analytics.log_approx(data)

"""Take sample data points"""
data_x = np.linspace(min(data[:,0]),max(data[:,0]),1001)
data_y = logapprox(data_x)
data = material_analytics.combine_data(data_x,data_y)

"""Determine average slope"""
av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0]) #rise over run
closest_index = lambda data, value: (np.abs(data-value)).argmin()
deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

"""Determine where slope is closest to average"""
bend = closest_index(deriv1st[:,1],av_slope)

"""Fitted this offset line to the left side"""
young_modulus = (data[bend,1]-data[0,1])/(data[bend,0]-data[0,0])
def linear_estimation(x):
    return data[0,1] + young_modulus*(x-offset)

"""Sample linear points"""
linear_y = linear_estimation(data_x)

"""Find closest point in fitted curve"""
difference_bw_est = np.abs(data_y-linear_y)
intersection = data[np.where(difference_bw_est==min(difference_bw_est))[0]]

"""Find closest point in original dataset"""
data = data_original
intersect_x = intersection[0,0]
intersect_index = closest_index(data[:,0],intersect_x)

"""Plot"""
plot.plotmult2D(data,data[intersect_index][None,])







data = parser.stress_strain('ref/HSRS/326').get_experimental_data()
data = material_analytics.adjust(data) # turns all nonreal values into 0

"""1st derivative of actual data"""
slopes = material_analytics.regularize(material_analytics.get_slopes(data))
deriv1st = material_analytics.combine_data(data[:,0],slopes)

"""Fit a log curve"""
logdata = material_analytics.log_prep(data, cutoff = 0)
logapprox = material_analytics.log_approx(logdata)

"""Take sample data points"""
data_x = np.linspace(min(logdata[:,0]),max(logdata[:,0]),1001)
data_y = logapprox(data_x)
logdata = material_analytics.combine_data(data_x,data_y)

"""1st derivative of the fitted log function"""
deriv1stlog = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(logdata))

"""First derivative vs first derivative of the logarithm"""
plot.plotmult2D(deriv1st, deriv1stlog, title = 'Slope of Stress vs Strain', xtitle = "Strain ($\epsilon$)", ytitle= "Stress Slope ($\partial \sigma / \partial \epsilon$)" )









cutoff = 0.00
offset = 0.002

"""Get data"""
data_original = parser.stress_strain('ref/850.dat').get_experimental_data()

"""Fit a log curve"""
data = material_analytics.log_prep(data_original, cutoff = cutoff)
logapprox = material_analytics.log_approx(data)

"""Take sample data points"""
data_x = np.linspace(min(data[:,0]),max(data[:,0]),1001)
data_y = logapprox(data_x)
data = material_analytics.combine_data(data_x,data_y)

"""Determine average slope"""
av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0]) #rise over run
closest_index = lambda data, value: (np.abs(data-value)).argmin()
deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

"""Determine where slope is closest to average"""
bend = closest_index(deriv1st[:,1],av_slope)

"""Fitted this offset line to the left side"""
young_modulus = (data[bend,1]-data[0,1])/(data[bend,0]-data[0,0])
def linear_estimation(x):
    return data[0,1] + young_modulus*(x-offset)

"""Sample linear points"""
linear_y = linear_estimation(data_x)

"""Find closest point in fitted curve"""
difference_bw_est = np.abs(data_y-linear_y)
intersection = data[np.where(difference_bw_est==min(difference_bw_est))[0]]

"""Find closest point in original dataset"""
data = data_original
intersect_x = intersection[0,0]
intersect_index = closest_index(data[:,0],intersect_x)

"""Plot"""
plot.plotmult2D(data,data[intersect_index][None,])

reload(material_analytics)

dat1 = parser.stress_strain('ref/850.dat').get_experimental_data()
plot.plotmult2D(dat1,material_analytics.yield_stress_classic(dat1))













cutoff = 0.00
offset = 0.002

"""Get data"""
data = parser.stress_strain('ref/850.dat').get_experimental_data()

"""Determine average slope"""
av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0]) #rise over run
closest_index = lambda data, value: (np.abs(data-value)).argmin()
deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

"""Determine where slope is closest to average"""
bend = closest_index(deriv1st[:,1],av_slope)

"""Fitted this offset line to the left side"""
young_modulus = (data[bend,1]-data[0,1])/(data[bend,0]-data[0,0])
def linear_estimation(x):
    return data[0,1] + young_modulus*(x-offset)

"""Sample linear points"""
linear_y = linear_estimation(data[:,0])

"""Find closest point in fitted curve"""
difference_bw_est = np.abs(data[:,1]-linear_y)
intersection = data[np.where(difference_bw_est==min(difference_bw_est))[0]]

"""Find closest point in original dataset"""
intersect_x = intersection[0,0]
intersect_index = closest_index(data[:,0],intersect_x)

"""Plot"""
plot.plotmult2D(data,data[intersect_index][None,])

def yield_classic_unfitted(data, cutoff = 0.0, offset = 0.002):

    """Determine average slope"""
    av_slope = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0])
    closest_index = lambda data, value: (np.abs(data-value)).argmin()
    deriv1st = material_analytics.combine_data(data[:,0],material_analytics.get_slopes(data))

    """Determine where slope is closest to average"""
    bend = closest_index(deriv1st[:,1],av_slope)

    """Fitted this offset line to the left side"""
    young_modulus = (data[bend,1]-data[0,1])/(data[bend,0]-data[0,0])
    def linear_estimation(x):
        return data[0,1] + young_modulus*(x-offset)

    """Sample linear points"""
    linear_y = linear_estimation(data[:,0])

    """Find closest point in fitted curve"""
    difference_bw_est = np.abs(data[:,1]-linear_y)
    intersection = data[np.where(difference_bw_est==min(difference_bw_est))[0]]

    """Find closest point in original dataset"""
    intersect_x = intersection[0,0]
    intersect_index = closest_index(data[:,0],intersect_x)

    return data[intersect_index][None,]









data = parser.stress_strain('ref/HSRS/22').get_experimental_data()
plot.plotmult2D(data,material_analytics.yield_stress(data))

dat1 = parser.stress_strain('ref/900.dat').get_experimental_data()
plot.plotmult2D(dat1,material_analytics.yield_stress_classic_unfitted(dat1))













data = parser.stress_strain('ref/HSRS/222').get_experimental_data()
yld = material_analytics.yield_stress(data)

yldindex = np.where(data==yld)[0][0]
print yldindex
plot.plotmult2D(data[:yldindex],data[yldindex:], title = 'Stress vs Strain', xtitle = "Strain ($\epsilon$)", ytitle= "Stress ($\sigma$)")







data = parser.stress_strain('ref/900.dat').get_experimental_data()
material_analytics.yield_stress(data, cutoff = 0.0002, decreasingend=True)



reload(material_analytics)
import graph_suite as plot
import parser

data = parser.stress_strain('ref/900.dat').get_experimental_data()

plot.plot2D(data, marker = 'o',title = 'Stress v Strain',xtitle ='Strain ($\epsilon$)',ytitle='Stress ($\sigma$)')

model = material_analytics.stress_model(data, material_analytics.yield_stress_classic_fitted(data))

nums = np.zeros(2)

for val in domain:
    nums = np.vstack((nums, np.asarray(model(val))[None,:]))
    
nums = nums[1:]

plot.plotmult2D(nums,data, marker1 = '-', marker2 = 'o',title = 'Stress v Strain',xtitle ='Strain ($\epsilon$)',ytitle='Stress ($\sigma$)')



model = material_analytics.stress_model(data, material_analytics.yield_stress_classic_fitted(data))
fittedpts = material_analytics.samplepoints(model,[0.,1.],10000)
plot.plotmult2D(data,fittedpts,marker1 = 'o', marker2 = '-',title = 'Stress v Strain',xtitle ='Strain ($\epsilon$)',ytitle='Stress ($\sigma$)')





reload(material_analytics)
import graph_suite as plot
import parser

data = parser.stress_strain('ref/850.dat').get_experimental_data()
model = material_analytics.log_approx(data)
fittedpts = material_analytics.samplepoints(model,[0.,max(data[:,0])],10000)
plot.plotmult2D(data,fittedpts,marker1 = 'o', marker2 = '-',title = 'Stress v Strain',xtitle ='Strain ($\epsilon$)',ytitle='Stress ($\sigma$)')





import irreversible_stressstrain
#reload(irreversible_stressstrain)

from irreversible_stressstrain import StressStrain as strainmodel
import optimization_suite
#reload(optimization_suite)

import material_analytics
import graph_suite as plot

model = strainmodel('ref/HSRS/22')
data = model.get_experimental_data()

"""[0,1] is the first row, second column, which is the stress values"""
SS_stress = material_analytics.yield_stress(data)[0,1]

"""It returned the best fit"""
model_params = optimization_suite.minimize_suite(model.mcfunc, methods=['L-BFGS-B',], guess = [-150,1] ,SS_stress=SS_stress)

"""Plots the data versus the fitted data"""
plot.plotmult2D(data, model.irreversible_model(model_params,SS_stress), marker1 ='o',  marker2 = 'g-')









from matplotlib import pyplot as plt
import matplotlib

defaultfontsize = 30
matplotlib.rcParams.update({'font.size': defaultfontsize}) # default font size

def texOn():
    plt.rc('text', usetex=True)

texOn()
model = strainmodel('ref/HSRS/22')
data = model.get_experimental_data()
SS_stress = material_analytics.yield_stress(data)[0,1]

strain = data[:,0]
stress = data[:,1]

plt.ylabel("Stress ($\sigma$)")
plt.xlabel("Strain ($\epsilon$)")
plt.xlim([0,max(strain)])
plt.title('Stress vs Strain')

plt.plot(strain,stress, 'bo', label = 'Experimental Data')

model_params = [(-97.2243219614, 1.62465067804), (-97.226373943, 1.35675756514), (-97.2352796645, 57.1105312168) 
,(-272.261846785, 2.36548167553) 
,(-97.2095787659, 58.2472771029) ]

methods = ['Nelder-Mead', 'CG', 'L-BFGS-B', 'COBYLA', 'SLSQP']

"""
Nelder-Mead (-97.2243219614, 1.62465067804) 
CG  (-97.226373943, 1.35675756514) 
L-BFGS-B  (-97.2352796645, 57.1105312168) 
COBYLA  (-272.261846785, 2.36548167553) 
SLSQP  (-97.2095787659, 58.2472771029) 
"""

for index, model_param in enumerate(model_params):
    datacurrent = model.irreversible_model(model_param,SS_stress)
    
    plt.plot(datacurrent[:,0],datacurrent[:,1],'-', label = methods[index] )

plt.legend(prop={'size':10})
plot.plot2D(data)

