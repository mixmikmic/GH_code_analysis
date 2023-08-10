import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.arange(30, 100, 0.1)
## LINEAR
# Create the membership functions
x_cold_lin = fuzz.trimf(x, [30, 30, 50])
x_mild_lin = fuzz.trimf(x, [30, 50, 70])
x_warm_lin = fuzz.trimf(x, [50, 70, 100])
x_hot_lin = fuzz.trimf(x, [70, 100, 100])

# Plot the results of the linear fuzzy membership
plt.figure()
plt.plot(x, x_cold_lin, 'b', linewidth=1.5, label='Cold')
plt.plot(x, x_mild_lin, 'k', linewidth=1.5, label='Mild')
plt.plot(x, x_warm_lin, 'm', linewidth=1.5, label='Warm')
plt.plot(x, x_hot_lin, 'r', linewidth=1.5, label='Hot')
plt.title('Temperature, Linear Fuzzy')
plt.ylabel('Membership')
plt.xlabel('Temperature (Fahrenheit)')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);

## GAUSSIAN 
# Create the membership functions
x_cold_gauss = fuzz.gaussmf(x, 30, 8)
x_mild_gauss = fuzz.gaussmf(x, 50, 8)
x_warm_gauss = fuzz.gaussmf(x, 70, 12)
x_hot_gauss = fuzz.gaussmf(x, 100, 8)

# Plot the results of the gaussian fuzzy membership
plt.figure()
plt.plot(x, x_cold_gauss, 'b', linewidth=1.5, label='Cold')
plt.plot(x, x_mild_gauss, 'k', linewidth=1.5, label='Mild')
plt.plot(x, x_warm_gauss, 'm', linewidth=1.5, label='Warm')
plt.plot(x, x_hot_gauss, 'r', linewidth=1.5, label='Hot')
plt.title('Temperature, Gaussian Fuzzy')
plt.ylabel('Membership')
plt.xlabel('Temperature')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);

# Plot to show the sum is not always 1
x_sum = x_cold_gauss + x_mild_gauss +         x_warm_gauss + x_hot_gauss
plt.figure()
plt.plot(x, x_sum, 'y', linewidth=1.5, label='Total')
plt.title('Temperature, Gaussian Fuzzy Sum')
plt.ylabel('Membership')
plt.xlabel('Temperature')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);

## NORMALIZED GAUSSIAN
# rescale
x_sum = x_cold_gauss + x_mild_gauss +         x_warm_gauss + x_hot_gauss
x_cold_rescale = x_cold_gauss / x_sum
x_mild_rescale = x_mild_gauss / x_sum
x_warm_rescale = x_warm_gauss / x_sum
x_hot_rescale = x_hot_gauss / x_sum      

# Plot the results of the rescaled gaussian fuzzy membership
plt.figure()
plt.plot(x, x_cold_rescale, 'b', linewidth=1.5, label='Cold')
plt.plot(x, x_mild_rescale, 'k', linewidth=1.5, label='Mild')
plt.plot(x, x_warm_rescale, 'm', linewidth=1.5, label='Warm')
plt.plot(x, x_hot_rescale, 'r', linewidth=1.5, label='Hot')
plt.title('Temperature, Rescaled Gaussian Fuzzy')
plt.ylabel('Membership')
plt.xlabel('Temperature')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);

# Plot to show the sum is not always 1
x_sum = x_cold_rescale + x_mild_rescale +         x_warm_rescale + x_hot_rescale
plt.figure()
plt.plot(x, x_sum, 'y', linewidth=1.5, label='Total')
plt.title('Temperature, Rescaled Gaussian Fuzzy Sum')
plt.ylabel('Membership')
plt.xlabel('Temperature')
plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5),
          ncol=1, fancybox=True, shadow=True);

