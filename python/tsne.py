get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_distances

digits = load_digits()
digits.data.shape

sample = digits.data[:20].reshape(20, 8, 8)
fig, axes = plt.subplots(4, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(sample[i], cmap='binary')
    ax.set(xticks=[], yticks=[])

proj = TSNE(n_components=2).fit_transform(digits.data)  # this will take a while

fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap='viridis')
ax.axis('off')
for i in range(10):
    xtext, ytext = np.median(proj[digits.target == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=6, foreground='white'), PathEffects.Normal()])

data = np.vstack([digits.data[digits.target==i] for i in range(10)])
y = np.hstack([digits.target[digits.target==i] for i in range(10)])
proj_ordered = TSNE(n_components=2).fit_transform(data)

fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(proj_ordered[:, 0], proj_ordered[:, 1], c=y, cmap='viridis')
ax.axis('off')
for i in range(10):
    xtext, ytext = np.median(proj_ordered[y == i, :], axis=0)
    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=6, foreground='white'), PathEffects.Normal()])

