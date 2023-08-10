from menpowidgets import (visualize_pointclouds, visualize_landmarkgroups, visualize_landmarks, 
                          visualize_images, visualize_patches, plot_graph, save_matplotlib_figure, 
                          features_selection)

get_ipython().magic('matplotlib inline')
import menpo.io as mio
from menpo.landmark import (labeller, face_ibug_68_to_face_ibug_49, face_ibug_68_to_face_ibug_66, 
                            face_ibug_68_to_face_ibug_68_trimesh, face_ibug_68_to_face_ibug_68)
from menpo.feature import igo, hog, lbp

im1 = mio.import_builtin_asset.breakingbad_jpg()
im1 = im1.crop_to_landmarks_proportion(0.2)
labeller(im1, 'PTS', face_ibug_68_to_face_ibug_68)

im2 = mio.import_builtin_asset.einstein_jpg()
im2 = im2.crop_to_landmarks_proportion(0.2)
im2 = igo(im2, double_angles=True)
labeller(im2, 'PTS', face_ibug_68_to_face_ibug_49)

im3 = mio.import_builtin_asset.lenna_png()
im3 = im3.crop_to_landmarks_proportion(0.2)
im3 = hog(im3)

im4 = mio.import_builtin_asset.takeo_ppm()
im4 = im4.crop_to_landmarks_proportion(0.2)
labeller(im4, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)

im5 = mio.import_builtin_asset.tongue_jpg()
im5 = im5.crop_to_landmarks_proportion(0.2)
im5 = im5.as_greyscale()

im6 = mio.import_builtin_asset.menpo_thumbnail_jpg()

pointclouds = [im1.landmarks['PTS'].lms,
               im2.landmarks['face_ibug_49'].lms,
               im3.landmarks['LJSON'].lms,
               im4.landmarks['PTS'].lms,
               im5.landmarks['PTS'].lms]

visualize_pointclouds(pointclouds)

landmark_groups = [im1.landmarks['PTS'], 
                   im2.landmarks['face_ibug_49'], 
                   im3.landmarks['LJSON'],
                   im4.landmarks['PTS'],
                   im5.landmarks['PTS'],
                   im3.landmarks['LJSON']]

visualize_landmarkgroups(landmark_groups, browser_style='slider')

landmarks = [im1.landmarks, 
             im2.landmarks, 
             im3.landmarks, 
             im4.landmarks, 
             im5.landmarks, 
             im6.landmarks]

visualize_landmarks(landmarks, style='minimal')

images = [im1, im2, im3, im4, im5, im6]

visualize_images(images)

patches1 = im1.extract_patches_around_landmarks(group='PTS')
pc1 = im1.landmarks['PTS'].lms
patches2 = im2.extract_patches_around_landmarks(group='face_ibug_49')
pc2 = im2.landmarks['face_ibug_49'].lms
patches3 = im3.extract_patches_around_landmarks(group='LJSON')
pc3 = im3.landmarks['LJSON'].lms
patches4 = im4.extract_patches_around_landmarks(group='face_ibug_68_trimesh')
pc4 = im4.landmarks['face_ibug_68_trimesh'].lms
patches5 = im5.extract_patches_around_landmarks(group='PTS')
pc5 = im5.landmarks['PTS'].lms

patches = [patches1, patches2, patches3, patches4, patches5]
patch_centers = [pc1, pc2, pc3, pc4, pc5]

visualize_patches(patches, patch_centers)

feat = features_selection()

feat[0](im1).view(channels=[0, 1]);

import numpy as np

x_axis = [x * np.pi / 10.0 for x in range(21)]

y_sin = list(np.sin(x_axis))
y_cos = list(np.cos(x_axis))
y_axis = [y_sin, y_cos]

plot_graph(x_axis, y_axis, legend_entries=['sin', 'cos'])

renderer = im1.view_landmarks(group='PTS')

save_matplotlib_figure(renderer)

images[0].view_widget(figure_size=(6, 4))

images[-1].view_widget(figure_size=(6, 4))

pointclouds[0].view_widget(figure_size=(6, 4))

landmark_groups[0].view_widget(figure_size=(6, 4))

landmarks[0].view_widget(figure_size=(6, 4))

