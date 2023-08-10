# import libraries
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

sns.set(palette='hls', font_scale=1.5)

# set variables name
product = ('Burger', 'Pizza', 'Coke', 'Fry')

# set index(will be labeld to be 'product')
p_range = np.arange(len(product))

# sales range
sales = 10 * np.random.rand(len(product))
# error range
error = 0.5 * np.random.rand(len(product))

# xeer: error(on x axis) range, alpha: opacity
plt.barh(p_range, sales, xerr=error, alpha=0.6)
plt.yticks(p_range, product)
plt.xlabel('Sales(million $)')
plt.ylabel('Products')
plt.show()

# the number of bar chart groups
n_groups = 4

# sales and std range on '15
sales_15 = 10 * np.random.rand(len(product))
std_15 = 0.5 * np.random.rand(len(product))

# sales and std range on '16
sales_16 = 15 * np.random.rand(len(product))
std_16 = 0.8 * np.random.rand(len(product))

# index, bar_width, opacity
index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.6

# error bar color
error_config = {'ecolor' : '0.6'}


# sales_15 plot
plt.bar(index, sales_15, bar_width, alpha=opacity,
        yerr=std_15, error_kw=error_config,
        label='Sales on \'15')

# sales_16 plot
## sales_16 plot will be placed on x axis(index + bar_width')
plt.bar(index+bar_width, sales_16, bar_width, alpha=opacity,
        yerr=std_16, error_kw=error_config,
        label='Sales on \'16')
'''
# stacked bar chart
## on x axis(index), can stack bar chart with 'bottom' arg.
plt.bar(index, sales_16, bar_width, alpha=opacity,
        yerr=std_16, error_kw=error_config,
        bottom=sales_15 # set the bottom plot
        label='Sales on \'16')
'''

plt.xlabel('Product')
plt.ylabel('Sales(million $)')
plt.title('Product Sales on 2016 and 2017')

# set the label position on between two plots
plt.xticks(index+bar_width/2, product)

plt.legend()
plt.tight_layout()

plt.show()

x = np.random.randn(100)
y = np.random.randn(100)

# make points on coordinate(x, y)
plt.scatter(x, y)
plt.show()

# make points on coordinate(x, y) with style
plt.scatter(x, y,
            s=np.random.randint(10, 500, 100), # size
            c=np.random.randn(100), # color
            edgecolors='black') # edge color
plt.show()

# display image with array
x = np.random.rand(5, 5)
print(x)

plt.imshow(x)
plt.grid(False) # off grid display
plt.show()

# various method
methods = [None, 'none', 'nearest', 'bilinear', 'bicubic',
           'spline16', 'spline36', 'hanning', 'hamming',
           'hermite', 'kaiser', 'quadric', 'catrom',
           'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

fig, axes = plt.subplots(3, 6,
                         subplot_kw={'xticks':[], 'yticks':[]})

# axes.flat: returns the axes as 1-dimensional(flat) array
for ax, method in zip(axes.flat, methods):
    ax.imshow(x, interpolation=method)
    ax.grid(False)
    ax.set_title(method)

plt.show()

