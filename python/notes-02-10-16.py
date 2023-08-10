import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

t = np.linspace(-2*np.pi,2*np.pi,100)

# Create a figure with 2 rows and 1 columns and set the current subplot to 1 and plot y = sin(t)
plt.subplot(2,1,1)
y1 = np.sin(t)
plt.plot(t,y1)
plt.xlim([-2*np.pi,2*np.pi])
plt.ylim([-1,1])
plt.title('Since and Cosing Waves')

# Set the current subplot to 2 and plot y = cos(t)
plt.subplot(2,1,2)
y2 = np.cos(t)
plt.plot(t,y2)
plt.xlim([-2*np.pi,2*np.pi])
plt.ylim([-1,1]);

plt.figure(figsize=(15,10))
t = np.linspace(0,3,1000)
nrows = 2
ncols = 3
y = 0
for n in range(0,nrows*ncols):
    plt.subplot(nrows,ncols,n+1)
    y = y + np.sin(2*np.pi*(2*n+1)*t)/(2*n+1)
    plt.plot(t,y)
    plt.ylim([-1,1])
    plt.title('Square Wave N = ' + str(n))

n = 100
samples = np.random.randn(n,2)
colors = np.random.randn(n)
sizes = np.random.randint(50,151,n)
plt.scatter(samples[:,0],samples[:,1],s=sizes,c=colors)
plt.xlim([-4,4])
plt.ylim([-4,4]);

price_data = np.genfromtxt('https://www.quandl.com/api/v3/datasets/BP/CRUDE_OIL_PRICES.csv',skip_header=1,delimiter=',',usecols=2)

price_data[:5]

plt.plot(price_data[::-1]);

plt.hist(price_data,bins=20);

get_ipython().system(' rm -r www.quandl.com')

