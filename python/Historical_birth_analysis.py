from __future__ import print_function,division,generators
from IPython.display import Image, display, display_png, HTML
get_ipython().magic('pylab inline')
import pandas as pd
import scipy.stats as stats

Adults = pd.read_csv('http://www.files.benlaken.com/documents/Adults.csv')    # Very nice!
Kids = pd.read_csv('http://www.files.benlaken.com/documents/Kids.csv')
Survival =pd.read_csv('http://www.files.benlaken.com/documents/Survival.csv')
DeathAge =pd.read_csv('http://www.files.benlaken.com/documents/age_death.csv')
Solar = pd.read_csv('http://www.files.benlaken.com/documents/Usoskin.csv')     # Read solar dataset also
Solar.index = Solar.Year                    # Index the solar data by year
#Adults.index = Adults.Born                 # Index the data by the year of birth
#Adults                                     # Preview the data

# Construct some summary data of number of people born per year

trange=arange(min(Adults.Born),max(Adults.Born))  # Generate a list of all years of data

BirthsPerYr = [len(Adults.Born[Adults.Born == year]) for year in trange] # How many adults born each year?

BirthsPerYr_GA = [len(Adults.Born[(Adults.Born == year) & (Adults.Group == 'A')])
                  for year in trange]                      
BirthsPerYr_GB = [len(Adults.Born[(Adults.Born == year) & (Adults.Group == 'B')])
                  for year in trange] 

BirthsPerYr = pd.Series(data=BirthsPerYr,index=trange)   # Make these pd series objects
BirthsPeYr_GA = pd.Series(data=BirthsPerYr_GB,index=trange)
BirthsPerYr_GB = pd.Series(data=BirthsPerYr_GB,index=trange)

fig1 = plt.figure()
fig1.set_size_inches(10,5) 
ax1 = fig1.add_subplot(111) 
ax1.plot(trange,BirthsPerYr,'xb')
ax1.set_xlim(min(trange),max(trange))
ax1.set_xlabel('Year')
ax1.set_ylabel(r'Births Yr$^{-1}$')
ax1.set_title(r'Recorded Historical births by Year in Adult dataset')
plt.grid(True)
plt.show(fig1)

fig2 = plt.figure()
fig2.set_size_inches(10,5) 
ax1 = fig2.add_subplot(111) 

width = 1.0       # the width of the bars: can also be len(x) sequence
ax1.bar(trange,BirthsPerYr_GA,   width, color='r',label='GroupA')
ax1.bar(trange,BirthsPerYr_GB, width, color='y',
             bottom=BirthsPerYr_GA,label='Group B')
ax1.set_xlim(min(trange),max(trange))
ax1.set_ylabel(r'Births Yr$^{-1}$')
ax1.set_title('Births per year by Group')
ax1.set_xlabel('Year')
ax1.legend()
plt.show(fig2)

# Old fashioned plot
fig3 = plt.figure()
fig3.set_size_inches(10,5) 
ax1 = fig3.add_subplot(111) 
ax1.plot(Solar.index[0:36],Solar.Counts[0:36],'-b') # Jump in data (not handeld well by plot)
ax1.plot(Solar.index[36:],Solar.Counts[36:],'-b')
ax1.set_xlabel('Year')
ax1.set_xlim(1675,1880)
ax1.set_ylabel(r'Neutron Counts $\times10^{-3} hr^{-1}$')
ax1.set_title(r'Reconstructed Solar Activity from Usoskin et al.(2002) over study period')
plt.grid(True)
plt.show(fig1)

# Load an enhanced copy of Figure 2 from Skjaervo et al. (2014)
# The significant fertility result in low-status (poor) females will be tested first.
Image(url='http://www.files.benlaken.com/documents/Norway_solar_fig2.png')

print('Adult Men with >0 Kids:',len(Adults.NumKids[(Adults.Gender == 'M')&(Adults.NumKids > 0)]))
print('Adult Women with >0 Kids:',len(Adults.NumKids[(Adults.Gender == 'F')&(Adults.NumKids > 0)]))

def GenFertilityStats(Gender,Sphase):
    '''Function to convienently generate a dictionary object of statistics.
    Gender can be either M or F
    Sphase can be either MAX or MIN
    This will return a dictionary of mean, count, std, and s.e. statistics for Fertility.
    '''
    tmp={}
    stats = ['count','mean','std','se']
    sep='_'
    for metric in stats:
        if metric == 'mean':
            tmp[metric]=(mean(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)]))
        if metric == 'count':
            tmp[metric]=(len(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)]))
        if metric == 'std':
            tmp[metric]=(std(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)]))
        if metric == 'se':
            tmp[metric]=(tmp['std']/np.sqrt(tmp['count']-1))
    return tmp


def GenFertilityStats_byClass(Gender,Sphase,Class):
    '''Function to convienently generate a dictionary object of statistics.
    Gender can be either M or F
    Sphase can be either MAX or MIN
    Class can be either High or Low (SocialStatus)
    This will return a dictionary of mean, count, std, and s.e. statistics for Fertility.
    '''
    tmp={}
    stats = ['count','mean','std','se']
    sep='_'
    for metric in stats:
        if metric == 'mean':
            tmp[metric]=(mean(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)
                                                                        &(Adults.SocialStatus == Class)]))
        if metric == 'count':
            tmp[metric]=(len(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)
                                                                       &(Adults.SocialStatus == Class)]))
        if metric == 'std':
            tmp[metric]=(std(Adults.NumKids[(Adults.Gender == Gender)&(Adults.SolarPhase == Sphase)
                                                                       &(Adults.SocialStatus == Class)]))
        if metric == 'se':
            tmp[metric]=(tmp['std']/np.sqrt(tmp['count']-1))
    return tmp

print('Fertility Summary statistics')
M_SMAX_fertility = GenFertilityStats(Gender='M',Sphase='MAX')
M_SMIN_fertility = GenFertilityStats(Gender='M',Sphase='MIN')
F_SMAX_lowclass = GenFertilityStats_byClass(Gender='F',Sphase='MAX',Class='Low')
F_SMIN_lowclass = GenFertilityStats_byClass(Gender='F',Sphase='MIN',Class='Low')
F_SMAX_highclass = GenFertilityStats_byClass(Gender='F',Sphase='MAX',Class='High')
F_SMIN_highclass = GenFertilityStats_byClass(Gender='F',Sphase='MIN',Class='High')

print('M_SMAX_fertility',M_SMAX_fertility)
print('M_SMIN_fertility',M_SMIN_fertility)
print('F_SMAX_lowclass',F_SMAX_lowclass)
print('F_SMIN_lowclass',F_SMIN_lowclass)
print('F_SMAX_highclass',F_SMAX_highclass)
print('F_SMIN_highclass',F_SMIN_highclass)

data = [
    Adults.NumKids[(Adults.Gender == 'M')&(Adults.SolarPhase == 'MAX')],
    Adults.NumKids[(Adults.Gender == 'M')&(Adults.SolarPhase == 'MIN')],
    Adults.NumKids[(Adults.Gender == 'F')&(Adults.SolarPhase == 'MAX')&(Adults.SocialStatus == 'High')],
    Adults.NumKids[(Adults.Gender == 'F')&(Adults.SolarPhase == 'MIN')&(Adults.SocialStatus == 'High')],
    Adults.NumKids[(Adults.Gender == 'F')&(Adults.SolarPhase == 'MAX')&(Adults.SocialStatus == 'Low')],
    Adults.NumKids[(Adults.Gender == 'F')&(Adults.SolarPhase == 'MIN')&(Adults.SocialStatus == 'Low')],
        ]

labels = list(['Males \n n=349',
               'Males \n n=1369',
               'Rich Females \n n=239',
               'Rich Females \n n=983',
               'Poor Females \n n=125',
               'Poor Females \n n=496'])

fig5 = plt.figure()
fig5.set_size_inches(15,8) 
ax1 = fig5.add_subplot(111) 
box = ax1.boxplot(data,notch=True,sym='b+',vert=True,bootstrap=10000,patch_artist=True)

colors = ['#FFBBA9', '#33D6FF', '#FFBBA9', '#33D6FF', '#FFBBA9','#33D6FF']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_ylim(-1,16)
ax1.set_title('Num of Children born i.e. Fertility (Red = Solar Max, Blue = Solar Min)  \n Bootstrapped 95% asymetric C.I.')
ax1.set_xlabel('Catagory')
ax1.set_ylabel(r'Fertility (N$^{o}$ Children)')
ax1.set_xticklabels(labels)
plt.show(fig5)
#plt.savefig("boxplots.png",dpi=100,transparent=True)

# Small sanity check: Number of children born for Men and Women should be approx equal as they should be the same children
print('Total Children of Men:',len(Adults.NumKids[(Adults.Gender == 'M')]))
print('Total Children of Women:',len(Adults.NumKids[(Adults.Gender == 'F')]))
print('Unclaimed Children:',len(Adults.NumKids[(Adults.Gender == 'F')]) - len(Adults.NumKids[(Adults.Gender == 'M')]))

NumKids_poor_females = Adults.NumKids[(Adults.Gender == 'F')&(Adults.SocialStatus == 'Low')]

MC_size = 100000
pop_n125 = []
pop_n496 = []
pop_n125=[mean(random.choice(NumKids_poor_females,size=125)) for n in range(MC_size)] # Yep, it's just that easy.
pop_n496=[mean(random.choice(NumKids_poor_females,size=496)) for n in range(MC_size)] # A one line MC. Hooray Python!

pop_125_hist, bins = np.histogram(pop_n125, bins=arange(3.0, 5.0, 0.025))  # Bin the MC data to generate a PDF
pop_496_hist, bins = np.histogram(pop_n496, bins=arange(3.0, 5.0, 0.025))  # will do this manually for more flexibility

# -- Calculate the PDFs for the two MC samples and create a step plot with Gaussian fit---
density_125 = pop_125_hist/sum(pop_125_hist)       # Normalize the population to 1.0
step = (bins[1] - bins[0]) *0.5                    # need to center the bins:
bcenter = [ bpos + step for bpos in bins[:-1]]     
width = bins[1] - bins[0]
xx = np.ravel(zip(bins[0:-1], bins[0:-1] + width))
yy = np.ravel(zip(density_125, density_125))

density_496 = pop_496_hist/sum(pop_496_hist)
yy2 = np.ravel(zip(density_496, density_496))

# Create ideal Gaussian distributions 
mu, sigma = np.mean(pop_n125), np.std(pop_n125)        # Idealized Gaussian...
ygauss = normpdf(bins, mu, sigma)                      # ...a np function to generate a Gaussian pdf
ynormgauss = ygauss/sum(ygauss)                        #...a normalized guassian function

mu2, sigma2 = np.mean(pop_n496), np.std(pop_n496)      # More clunky boiler plate code (I should function this)
ygauss2 = normpdf(bins, mu2, sigma2)                      
ynormgauss2 = ygauss2/sum(ygauss2)                        

print('Probability Density Functions of 100k samples from poor female data')
#-----Plotting-----
my_hist1 = plt.figure()
my_hist1.set_size_inches(12,5)            # Specify the output size
ax1 = my_hist1.add_subplot(121)              # Add an axis frame object to the plot (i.e. a pannel)
ax2 = my_hist1.add_subplot(122)              # Add an axis frame object to the plot (i.e. a pannel)

# first pannel
ax1.bar(bins[0:-1], density_125, width=width, facecolor='k',linewidth=0.0,alpha=0.2)  # Filled bars
ax1.plot(bins,ynormgauss,'r-',linewidth=1.5)                                          # Ideal Gaussian line
ax1.vlines(F_SMAX_lowclass['mean'], 0.0, 0.1,colors='b')                              # Marker line of Mean
ax1.grid(True)
ax1.set_ylabel(r'Density',fontsize=11)
ax1.set_xlabel('Mean fertility in random samples of n=125',fontsize=11)
leg1=ax1.legend(['Gaussian fit','Solar Max sample','MC population',],prop={'size':11},
                numpoints=1,markerscale=5.,frameon=True,fancybox=True)
leg1.get_frame().set_alpha(1.0)                # Make the ledgend semi-transparent
ax1.plot(xx, yy, 'k',alpha=0.8)                # Edges for the step plot
ax1.vlines(F_SMAX_lowclass['mean']+F_SMAX_lowclass['se'], 0.0, 0.1,colors='b',linestyle='dashed')
ax1.vlines(F_SMAX_lowclass['mean']-F_SMAX_lowclass['se'], 0.0, 0.1,colors='b',linestyle='dashed')
ax1.set_ylim(0.00,0.09)

# -- second pannel
ax2.bar(bins[0:-1], density_496, width=width, facecolor='k',linewidth=0.0,alpha=0.2)  # Filled bars
ax2.plot(bins,ynormgauss2,'r-',linewidth=1.5) # Ideal Gaussian line
ax2.vlines(F_SMIN_lowclass['mean'], 0.0, 0.1,colors='b')  
ax2.grid(True)
ax2.set_ylabel(r'Density',fontsize=11)
ax2.set_xlabel('Mean fertility in random samples of n=496',fontsize=11)
leg2=ax2.legend(['Gaussian fit','Solar Min sample','MC population',],prop={'size':11},
                numpoints=1,markerscale=5.,frameon=True,fancybox=True)
leg2.get_frame().set_alpha(1.0)                # Make the ledgend semi-transparent
ax2.plot(xx, yy2, 'k',alpha=0.8)                # Edges for the step plot
ax2.vlines(F_SMIN_lowclass['mean']+F_SMIN_lowclass['se'], 0.0, 0.1,colors='b',linestyle='dashed')
ax2.vlines(F_SMIN_lowclass['mean']-F_SMIN_lowclass['se'], 0.0, 0.1,colors='b',linestyle='dashed')
ax2.set_ylim(0.00,0.09)
#plt.show(my_hist1)
plt.savefig("Solar_norway_distrib.png",dpi=100,transparent=True)

# Use a percentile rank function to find the rank of a given value:
print('Means ±S.E. of Poor Female sample during Solar Maximum (n=125)')
print('F_SMAX_lowclass:',F_SMAX_lowclass['mean'],'±',F_SMAX_lowclass['se'])
print('Percentile rank compared to random (null) MC-cases:')
print('Percentile of mean:',stats.percentileofscore(pop_n125, F_SMAX_lowclass['mean'])) 
print('Percentile of lower CI:',stats.percentileofscore(pop_n125, F_SMAX_lowclass['mean']+F_SMAX_lowclass['se'])) 
print('Percentile of upper CI:',stats.percentileofscore(pop_n125, F_SMAX_lowclass['mean']-F_SMAX_lowclass['se'])) 

# Use a percentile rank function to find the rank of a given value:
print('Means ±S.E. of Poor Female sample during Solar Minimum (n=496)')
print('F_SMAX_lowclass:',F_SMIN_lowclass['mean'],'±',F_SMIN_lowclass['se'])
print('Percentile rank compared to random (null) MC-cases:')
print('Percentile of mean:',stats.percentileofscore(pop_n496, F_SMIN_lowclass['mean'])) 
print('Percentile of lower CI:',stats.percentileofscore(pop_n496, F_SMIN_lowclass['mean']+F_SMIN_lowclass['se'])) 
print('Percentile of upper CI:',stats.percentileofscore(pop_n496, F_SMIN_lowclass['mean']-F_SMIN_lowclass['se'])) 

