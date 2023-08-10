get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(sum(map(ord, "aesthetics")))

#simple function definition
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

#from matplotlib
sinplot()

import seaborn as sns
sinplot()

# five backgrounds
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)

sns.boxplot(data=data)
sns.despine()

sns.set() #reset previous options to default
f, ax = plt.subplots()
sns.violinplot(data)
sns.despine(offset=10, trim=True)

sns.set_style("whitegrid")
sns.boxplot(data=data, palette="deep")
sns.despine(left=True)

#Use with statement with axes_style for subplots with different styles
with sns.axes_style("darkgrid"):
    plt.subplot(121)
    sinplot()

plt.subplot(122)
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sinplot()

with sns.plotting_context("paper"):
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    sinplot()
plt.figure(figsize=(10,6))
plt.subplot(212)
sns.set_context("paper",font_scale=1.5, rc={"lines.linewidth": 2.5}) #font can change, others rc+dictionaries
sinplot(-2)
    
    

sns.set(rc={"figure.figsize": (6, 6)})
np.random.seed(sum(map(ord, "palettes")))

current_palette = sns.color_palette()
sns.palplot(current_palette)

sns.palplot(sns.color_palette("hls", 8)) #depending on how many colors you need, hue in circular

sns.palplot(sns.hls_palette( 8, l=0.2, s=0.6))

sns.palplot(sns.color_palette("husl", 8))

current_palette = sns.choose_colorbrewer_palette('qualitative') #for choosing different colors as required

sns.palplot(current_palette)

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"] #customized
sns.palplot(sns.color_palette(flatui))

#xkcd
sns.palplot(sns.xkcd_palette(['denim blue','faded green']))

#sequential
sns.palplot(sns.color_palette("BuGn"))

sns.palplot(sns.color_palette("BuGn_r")) #reverse

sns.palplot(sns.color_palette("BuGn_d")) #dark

sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)) #cubehelix, more options ALLINONE

x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.kdeplot(x, y, cmap=cmap, shade=True);

pal = sns.dark_palette("palegreen", as_cmap=True) #dark paleete light palette
sns.kdeplot(x, y, cmap=pal);

# default sns.set_palette() takes vlues like color_palette but sets it for whole notebook. 
#Use with statement for subplots...

tips = sns.load_dataset("tips")
g = sns.PairGrid(tips,
                 x_vars=["smoker", "time", "sex"],
                 y_vars=["total_bill", "tip"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot,palette="pastel");

#g.map(sns.swarmplot, palette="pastel");



