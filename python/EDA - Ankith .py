import pandas as pd
import cv2
import seaborn
import matplotlib.pyplot as plt
import numpy as np

limage = cv2.imread("mdb001.tif")
rimage = cv2.imread("mdb002.tif")

plt.figure(figsize =(8,8))
plt.subplot(1,2,1)
plt.imshow(limage)

plt.subplot(1,2,2)
plt.imshow(rimage)
plt.show()

miasminimeta = pd.read_csv("mias-mini-meta.csv")

miasminimeta.columns

miasminimeta.shape

miasminimeta['abnormaility_class'].value_counts()

miasminimeta['background_tissue_type'].value_counts()

miasminimeta['severity'].value_counts()

miasminimeta['radius'].value_counts().head(5)

center_x = [x.strip() for x in miasminimeta['center_x'] if type(x) is str]
center_y = [x.strip() for x in miasminimeta['center_y'] if type(x) is str]

xC = []
yC = []
for x in center_x:
    try:
        xC.append(int(x))
    except:
        pass
        
for x in center_y:
    try:
        yC.append(int(x))
    except:
        pass


get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(xC,yC)
plt.title('Center of Abnormality in terms of pixel')

get_ipython().system('ls')

mass_case =  pd.read_csv("mass_case_description_train_set.csv")

mass_case.columns

mass_case.shape

mass_case['mass_shape'].value_counts()

mass_case['assessment'].value_counts()

mass_case['mass_margins'].value_counts()

mass_case['pathology'].value_counts()

malignant_cases = mass_case[mass_case['pathology']=='MALIGNANT']

malignant_cases.shape

malignant_cases['mass_margins'].value_counts()

cc_views = mass_case[mass_case['view']=='CC']

cc_views['pathology'].value_counts()

mlo_views = mass_case[mass_case['view']=='MLO']

mlo_views['pathology'].value_counts()

mass_case['view'].value_counts()

