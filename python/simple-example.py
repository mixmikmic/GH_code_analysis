import matplotlib.pyplot as plt
import numpy as np
import inference

get_ipython().magic('matplotlib inline')

nb_items = 10
nb_rankings = 500
size_of_ranking = 3

strengths = np.exp(np.random.rand(nb_items))
rankings = inference.generate_rankings(strengths, nb_rankings, size_of_ranking)

rankings[:10]

# Spectral estimate using LSR.
spectral_estimate = inference.lsr(nb_items, rankings)

# ML estimate using I-LSR.
ml_estimate = inference.ilsr(nb_items, rankings)

def plot_strengths(ax, strengths):
    img = np.log(strengths)[np.newaxis,:]
    ax.imshow(img, interpolation='nearest', cmap=plt.get_cmap('YlGnBu'))
    ax.yaxis.set_visible(False)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10.0, 5.0))

plot_strengths(ax1, strengths)
ax1.set_title("True parameters")

plot_strengths(ax2, ml_estimate)
ax2.set_title("ML estimate")

plot_strengths(ax3, spectral_estimate)
ax3.set_title("Spectral estimate")

