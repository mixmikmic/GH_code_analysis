import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

norm_data = np.random.normal(0,1, 1000) #normally distributed data 0-1, 100 data points 

sm.qqplot(data=norm_data, line='45')
plt.show()

import seaborn as sb 
from scipy.stats import skewnorm

#create distributions 
norm_data = np.random.normal(loc=0, scale=1, size=1000)
left_skew = skewnorm.rvs(7, size=1000)
right_skew = skewnorm.rvs(-7, size=1000)
heavy_tail = np.random.standard_cauchy(size=1000)
heavy_tail = heavy_tail = heavy_tail[(heavy_tail>-25) & (heavy_tail<25)] #limit tails for plotting visuals
light_tail =  np.random.uniform(low=0, high=1, size=1000) #few extrema 
bimodal = np.concatenate((np.random.normal(loc=0, scale=1, size=500), #concat two normals w/ diff locations
                          np.random.normal(loc=10, scale=1, size=500)
                         ),axis=0)

#create dictionary for distributions 
sets = {'Normal Distribution':norm_data,
        'Left Skew':left_skew,
        'Right Skew':right_skew,
        'Heavy Tail Distribution':heavy_tail,
        'Light Tailed Distribution':light_tail,
        'Bimodal Distribution':bimodal}       

#loop and plot   
for dataset in sets:
    print dataset
    print 'Kernel Density Estimation, Histogram                             QQ Plot'
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    fig.set_size_inches(20,5)
    
    sb.distplot(sets[dataset], bins=25, ax=ax1) #KDE and histogram 
    sm.qqplot(data=sets[dataset], line='45', fit=True, ax=ax2) #sQQ plot
    plt.show()

