import matplotlib as mpl

mpl

# I normally prototype my code in an editor + ipy terminal.
# In those cases I import pyplot and numpy via
import matplotlib.pyplot as plt
import numpy as np

# In Jupy notebooks we've got magic functions and pylab gives you pyplot as plt and numpy as np
# %pylab 

# Additionally, inline will let you plot inline of the notebook
# %pylab inline

# And notebook, as I've just found out gives you some resizing etc... tools inline.
# %pylab notebook

y = np.ones(10)

for x in range(2,10):
    y[x] = y[x-2] + y[x-1]

plt.plot(y)
plt.title('This story')

plt.show()

print('I can not run this command until I close the window because interactive mode is turned off')

get_ipython().magic('pylab inline')

# Set default figure size for your viewing pleasure...
pylab.rcParams['figure.figsize'] = (10.0, 7.0)

x = np.linspace(0,5,100)
y = np.random.exponential(1./3., 100)

# Make a simply plot of x vs y, Set the points to have an 'x' marker.
plt.plot(x,y, c='r',marker='x')

# Label our x and y axes and give the plot a title.
plt.xlabel('Sample time (au)')
plt.ylabel('Exponential Sample (au)')
plt.title('See the trend?')

x = np.linspace(0,6.,1000.)

# Alpha = 0.5, color = red, linstyle = dotted, linewidth = 3, label = x
plt.plot(x, x, alpha = 0.5, c = 'r', ls = ':', lw=3., label='x')

# Alpha = 0.5, color = blue, linstyle = solid, linewidth = 3, label = x**(3/2)
# Check out the LaTeX!  
plt.plot(x, x**(3./2), alpha = 0.5, c = 'b', ls = '-', lw=3., label=r'x$^{3/2}$')

# And so on...
plt.plot(x, x**2, alpha = 0.5, c = 'g', ls = '--', lw=3., label=r'x$^2$')


plt.plot(x, np.log(1+x)*20., alpha = 0.5, c = 'c', ls = '-.', lw=3., label='log(1+x)')

# Add a legend (loc gives some options about where the legend is placed)
plt.legend(loc=2)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

# size = area variable, c = colors variable
x = plt.scatter(x, y, s=area, c=colors, alpha=0.4)
plt.show()

N=10000
values1 = np.random.normal(25., 3., N)
values2 = np.random.normal(33., 8., N/7)
valuestot = np.concatenate([values1,values2])

binedges = np.arange(0,101,1)
bincenters = (binedges[1:] + binedges[:-1])/2.

# plt.hist gives you the ability to histogram and plot all in one command.
x1 = plt.hist(valuestot, bins=binedges, color='g', alpha=0.5, label='total')
x2 = plt.hist(values2, bins=binedges, color='r', alpha=0.5, histtype='step', linewidth=3, label='values 1')
x3 = plt.hist(values1, bins=binedges, color='b', alpha=0.5, histtype='step', linewidth=3, label='values 2')

plt.legend(loc=7)

fig = plt.figure(figsize=(10,6))

# Make an axes as if the figure had 1 row, 2 columns and it would be the first of the two sub-divisions.
ax1 = fig.add_subplot(121)
plot1 = ax1.plot([1,2,3,4,1,0])
ax1.set_xlabel('time since start of talk')
ax1.set_ylabel('interest level')
ax1.set_xbound([-1.,6.])



# Make an axes as if the figure had 1 row, 2 columns and it would be the second of the two sub-divisions.
ax2 = fig.add_subplot(122)
plot2 = ax2.scatter([1,1,1,2,2,2,3,3,3,4,4,4], [1,2,3]*4)
ax2.set_title('A commentary on chairs with wheels')

print(plot1)
print(plot2)

fig2 = plt.figure(figsize=(10,10))
ax1 = fig2.add_axes([0.1,0.1,0.8,0.4])

histvals = ax1.hist(np.random.exponential(0.5,5000), bins=np.arange(0,5, 0.1))

ax1.set_xlabel('Sampled Value')
ax1.set_ylabel('Counts per bin')

ax2 = fig2.add_axes([0.3,0.55, 0.7, 0.45])
ax2.plot([13,8,5,3,2,1,1],'r:',lw=3)

import scipy.stats as stats

# With subplots we can make all of the axes at ones.  
# The axes are return in a list of lists. 
f, [[ax0, ax1], [ax2, ax3]] = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)

# Remove the space between the top and bottom rows of plots
# wspace would do the same for left and right columns...
f.subplots_adjust(hspace=0)

ax0.plot(range(50,250), np.exp(np.arange(50,250) / 23.) )
ax2.scatter(np.random.normal(125,27,100), np.random.binomial(200,0.4,100))

ax1.plot(range(0,300), np.random.exponential(0.5,300), 'g')
ax3.plot(range(0,300), stats.norm.pdf(np.arange(0,300),150, 30) , 'g')

plt.colormaps()

cmap0 = plt.cm.cubehelix
cmap1 = plt.cm.Accent
cmap2 = plt.cm.Set1
cmap3 = plt.cm.Spectral

colmaps = [cmap0,cmap1,cmap2,cmap3]

Ncolors = 12
col0 = cmap0(np.linspace(0,1,Ncolors))

f, [[ax0, ax1], [ax2, ax3]] = plt.subplots(nrows=2, ncols=2, figsize=(13,13))
x = np.linspace(0.01,100,1000)

for idx, axis in enumerate([ax0,ax1,ax2,ax3]):
    colormap = colmaps[idx]
    colors = colormap(np.linspace(0,1,Ncolors))

    axis.set_title(colormap.name)
    
    for val in range(Ncolors):
        axis.plot(x,x**(1.0 + 0.1 * val), c=colors[val], lw=3, label=val)
    axis.loglog()

# Lets look at a two distributions on an exponential noise background...
Nnoise = 475000
Nnorm1 = 10000
Nnorm2 = 15000

# Uniform noise in x, exponential in y
xnoise = np.random.rand(Nnoise) * 100
ynoise = np.random.exponential(250,475000)

# Uniform in X, normal in Y
xnorm1 = np.random.rand(Nnorm1) * 100
ynorm1 = np.random.normal(800, 50, Nnorm1)

# Normal in X and Y
xnorm2 = np.random.normal(50, 30, 15000)
ynorm2 = np.random.normal(200, 25, 15000)

xtot = np.concatenate([xnoise, xnorm1, xnorm2])
ytot = np.concatenate([ynoise, ynorm1, ynorm2])

xbins = np.arange(0,100,10)
ybins = np.arange(0,1000,10)
H, xe, ye = np.histogram2d(xtot, ytot, bins=[xbins, ybins])

X,Y = np.meshgrid(ybins,xbins)

fig4 = plt.figure(figsize=(13,8))
ax1 = fig4.add_axes([0.1,0.1,0.35,0.4])
ax2 = fig4.add_axes([0.5,0.1,0.35,0.4])

pcolplot = ax1.pcolor(X, Y, H, cmap=cm.GnBu)
ax1.set_title('Linear Color Scale')
plt.colorbar(pcolplot, ax=ax1)

from matplotlib.colors import LogNorm
pcolplot2 = ax2.pcolor(X, Y, H, norm=LogNorm(vmin=H.min(), vmax=H.max()), cmap=cm.GnBu)
ax2.set_title('Log Color Scale')
plt.colorbar(pcolplot2, ax=ax2)

xvals = np.arange(0,120,0.1)

# Define a few functions to use
f1 = lambda x: 50. * np.exp(-x/20.)
f2 = lambda x: 30. * stats.norm.pdf(x, loc=25,scale=5) 
f3 = lambda x: 200. * stats.norm.pdf(x,loc=40,scale=10) 
f4 = lambda x: 25. * stats.gamma.pdf(x, 8., loc=45, scale=4.)

# Normalize to define PDFs
pdf1 = f1(xvals) / (f1(xvals)).sum()
pdf2 = f2(xvals) / (f2(xvals)).sum()
pdf3 = f3(xvals) / (f3(xvals)).sum()
pdf4 = f4(xvals) / (f4(xvals)).sum()

# Combine them and normalize again
pdftot = pdf1 + pdf2 + pdf3 + pdf4
pdftot = pdftot / pdftot.sum()

fig5 = plt.figure(figsize=(11,8))
ax3 = fig5.add_axes([0.1,0.1,0.9,0.9])

# Plot the pdfs, and the total pdf
lines = ax3.plot(xvals, pdf1,'r', xvals,pdf2,'b', xvals,pdf3,'g', xvals,pdf4,'m')
lines = ax3.plot(xvals, pdftot, 'k', lw=5.)

# Calculate the mean
mean1 = (xvals * pdf1).sum()
mean2 = (xvals * pdf2).sum()
mean3 = (xvals * pdf3).sum()
mean4 = (xvals * pdf4).sum()

fig6 = plt.figure(figsize=(11,8))
ax4 = fig6.add_axes([0.1,0.1,0.9,0.9])

# Plot the total PDF
ax4.plot(xvals, pdftot, 'k', lw=5.)

# Grabe the limits of the y-axis for defining the extent of our vertical lines
axmin, axmax = ax4.get_ylim()

# Draw vertical lines. (x location, ymin, ymax, color, linestyle)
ax4.vlines(mean1, axmin, axmax, 'r',':')
ax4.vlines(mean2, axmin, axmax, 'b',':')
ax4.vlines(mean3, axmin, axmax, 'g',':')
ax4.vlines(mean4, axmin, axmax, 'm',':')

# Add some text to figure to describe the curves
# (xloc, yloc, text, color, fontsize, rotation, ...)
ax4.text(mean1-18, 0.0028, r'mean of $f_1(X)$', color='r', fontsize=18)
ax4.text(mean2+1, 0.0005, r'mean of $f_2(X)$', color='b', fontsize=18)
ax4.text(mean3+1, 0.0002, r'mean of $f_3(X)$', color='g', fontsize=18)
ax4.text(mean4+1, 0.0028, r'mean of $f_4(X)$', color='m', fontsize=18, rotation=-25)
temp = ax4.text(50, 0.0009, r'$f_{tot}(X)$', color='k', fontsize=22)

# Compute CDFs
cdf1 = pdf1.cumsum()
cdf2 = pdf2.cumsum()
cdf3 = pdf3.cumsum()
cdf4 = pdf4.cumsum()
cdftot = pdftot.cumsum()

fig7 = plt.figure(figsize=(11,8))
ax7 = fig7.add_axes([0.1,0.1,0.9,0.9])

# Plot them
ax7.plot(xvals, cdftot, 'k', lw=3)
ax7.plot(xvals, cdf1, 'r', ls=':', lw=2)
ax7.plot(xvals, cdf2, 'b', ls=':', lw=2)
ax7.plot(xvals, cdf3, 'g', ls=':', lw=2)
ax7.plot(xvals, cdf4, 'm', ls=':', lw=2)

# Force the y limits to be (0,1)
ax7.set_ylim(0,1.)

# Add 50% and 90% lines.
ax7.hlines(0.5, 0, 120., 'k', '--', lw=2)
ax7.hlines(0.95, 0, 120., 'k', '--', lw=3)

# Add some text
ax7.set_title('CDFs of dists 1-4 and total with 50% and 95% bounds')
ax7.text(110, 0.46, r'$50\%$ ', color='k', fontsize=20)
temp = ax7.text(110, 0.91, r'$95\%$ ', color='k', fontsize=20)

import matplotlib.image as mpimg

img=mpimg.imread('Tahoe.png')

imgplot = plt.imshow(img)

f, [ax0,ax1,ax2] = plt.subplots(nrows=3, ncols=1, figsize=(10,15))
f.subplots_adjust(hspace=0.05)

for ax in [ax0,ax1,ax2]:
#     ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    
ax0.imshow(img[:,:,0], cmap=cm.Spectral)
ax1.imshow(img[:,:,1], cmap=cm.Spectral)
ax2.imshow(img[:,:,2], cmap=cm.Spectral)

