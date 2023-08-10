import SimpleITK as sitk
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

image = sitk.ReadImage('../data/dot.dcm')

image.GetSize()

array = sitk.GetArrayFromImage(image)
print('`array` has type: {}'.format(type(array)))

array.shape

array[0, 0:10, 0:10, 1]

fig = plt.figure(figsize=(6, 6))
plt.imshow(array[0]);



