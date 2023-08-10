get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns

from neurosynth.base.dataset import Dataset
dataset = Dataset.load("data/neurosynth_60_0.4.pkl")

from coactivation import coactivation_contrast
from plotting import make_thresholded_slices

cut_coords = np.arange(-15, 60, 12) # I use the same z coordinates for each axial plot

contrast_mas = coactivation_contrast(
    dataset, 'images/cluster_labels_k3.nii.gz', q = 0.001)
make_thresholded_slices(contrast_mas, sns.color_palette('Set1', 3), 
                       cut_coords=cut_coords, annotate=False)

# Here I define the groupings
posterior = [3, 6]
middle = [1, 5, 7, 9]
anterior = [2, 4, 8]

from plotting import nine_colors
contrast_mas = coactivation_contrast(
    dataset, 'images/cluster_labels_k9.nii.gz', posterior)
colors = list(reversed([c for i, c in enumerate(nine_colors) if i + 1 in posterior]))
make_thresholded_slices(list(reversed(contrast_mas)), colors, 
                       cut_coords=cut_coords, annotate=False)

make_thresholded_slices(list(reversed(contrast_mas)), colors, 
                       cut_coords=range(-60, 50, 18), display_mode='y', annotate=False)

contrast_mas = coactivation_contrast(
    dataset, 'images/cluster_labels_k9.nii.gz', middle)
colors = [c for i, c in enumerate(nine_colors) if i + 1 in middle]
make_thresholded_slices(contrast_mas, colors, 
                       cut_coords=cut_coords, annotate=False)

make_thresholded_slices(contrast_mas, colors, cut_coords=range(-60, 50, 18), display_mode='y', annotate=False)

contrast_mas = coactivation_contrast(
    dataset, 'images/cluster_labels_k9.nii.gz', anterior)
colors = [c for i, c in enumerate(nine_colors) if i + 1 in anterior]
make_thresholded_slices(contrast_mas, colors, 
                       cut_coords=cut_coords, annotate=False)

make_thresholded_slices(contrast_mas, colors, cut_coords=range(-60, 50, 18), 
                        display_mode='y', annotate=False)

