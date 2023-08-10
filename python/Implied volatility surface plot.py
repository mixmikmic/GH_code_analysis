import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# generate strike prices between 100 and 200

strike_price = np.linspace(100,200,50)
time_to_maturity = np.linspace(0.25,3,50)

# build a coordinate system with 'x' and 'y' variables
strike_price, time_to_maturity = np.meshgrid(strike_price, time_to_maturity)

# generate pseudo-implied volatility by using strike price and time-to-maturity as parameters
implied_vol = ((strike_price - 150)**2)/(150 * strike_price)/(np.power(time_to_maturity, 0.95))

len(implied_vol[0])

fig = plot.figure(figsize = (10,5)) ## a plot object
ax = Axes3D(fig) # create a 3D object/handle

##plot surface: array row/column stride(step size:2)
surf = ax.plot_surface(strike_price, time_to_maturity, implied_vol, rstride = 2, cstride = 2, cmap = cm.coolwarm, linewidth = 0.5, antialiased = False)

#set x,y,a labels
ax.set_xlabel('Strike Price')
ax.set_ylabel('time to maturity')
ax.set_zlabel('implied volatility')
plot.show()

from IPython.display import Image
Image('figure_1.png')



from scipy import stats

def BSM_vega(S,K,T,r,sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T)/ (sigma * np.sqrt(T))
    N_d1 = stats.norm.cdf(d1, 0.0, 1.0)
    
    N_d1_derivative = stats.norm.pdf(d1, 0.0, 1.0)*np.sqrt(T)
    vega = S * N_d1_derivative * np.sqrt(T)
    return vega

S = 80
K = 100
T = 0.75
r = 0.105
sigma = 0.1

BSM_vega(S,K,T,r,sigma)

