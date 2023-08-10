get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ggplot import *

heart = pd.read_csv("http://faculty.washington.edu/kenrice/heartgraphs/nhaneslarge.csv", sep=",", na_values=".")

ggplot(heart, aes(x = 'DR1TFOLA')) +  geom_histogram()

ggplot(heart, aes(x = 'DR1TFOLA')) +  geom_histogram() +  labs(x = "Folate intake")

ggplot(heart, aes(x = 'DR1TFOLA')) +  geom_histogram(color = "white") +  labs(x = "Folate intake")

ggplot(heart, aes(x = 'DR1TFOLA')) +  geom_histogram(color = "white", fill = "peachpuff") +  labs(x = "Folate intake")

ggplot(heart, aes(x = "DR1TFOLA")) +  geom_histogram(color = "white", fill = "peachpuff") +  labs(x = "Folate intake") +  facet_grid(y="gender")

ggplot(heart, aes(x = "DR1TFOLA")) +  geom_density(color = "black") +  labs(x = "Folate intake") +  facet_grid(y="gender")

ggplot(heart, aes(x = "DR1TFOLA", color = "gender")) +  geom_density() +  labs(x = "Folate intake")

# ggplot(heart, aes(x = "gender", y = "BPXSAR")) +\
#  geom_point()

ggplot(heart, aes(x = "RIDAGEYR", y = "BPXSAR")) +  geom_point() +  labs(x = "Age (years)", y = "Systolic BP (mmHg)")

ggplot(heart, aes(x = "RIDAGEYR", y = "BPXSAR")) +  geom_point() +  stat_smooth(method='lm')+  labs(x = "Age (years)", y = "Systolic BP (mmHg)")

ggplot(heart, aes(x = "RIDAGEYR", y = "BPXSAR")) +  geom_point() +  stat_smooth(method='ma', window=5) +  labs(x = "Age (years)", y = "Systolic BP (mmHg)")

ggplot(heart, aes(x = "RIDAGEYR", y = "BPXSAR")) +  geom_point() +  stat_smooth(method='loess')+  labs(x = "Age (years)", y = "Systolic BP (mmHg)")

heart = heart.assign(age_cat = pd.cut(heart['RIDAGEYR'],[0,30,55,100]).astype('category'))
# heart = heart.dropna()

ggplot(heart, aes(x = "BMXBMI", y = "BPXSAR")) +  geom_point() +  facet_grid(y="age_cat") +  labs(x = "Body Mass Index", y = "Systolic BP (mmHg)")

ggplot(heart, aes(x = "BMXBMI", y = "BPXSAR")) +  geom_point() +  facet_grid(x="gender", y="age_cat") +  labs(x = "Body Mass Index", y = "Systolic BP (mmHg)")

