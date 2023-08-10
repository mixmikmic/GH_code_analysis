import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

n_timepoints = 100
onsets = [5, 13, 26]
duration = 6 # timepoints, unclear what unit
x = np.zeros(n_timepoints)
for on in onsets:
    x[on:on+duration] = 1
plt.plot(x)

def define_indicator(n_timepoints, onsets, duration):
    x = np.zeros(n_timepoints)
    for on in onsets:
        x[on:on+duration] = 1
    return x

n_timepoints

duration = 2
Xwow = define_indicator(n_timepoints, onsets, duration)
plt.plot(Xwow)

[1,2,3] * 3

np.array([1,2,3]) * 3

my_strings = ['my','name','is','mark']
for yo in my_strings:
    print(yo)
    print('.')

for x in np.linspace(0,10,20):
    print(x)

for x in np.arange(0,10,2):
    print(x)

for x in [1,2,3]:
    print(x**2)

# Do this by indexing into the array:
x_categorical = np.zeros((100,))
x_categorical[10:20] = 1
plt.plot(x_categorical);

class_list = ['Ana Chkhaidze', 
              'Michael Compton', 
              'Kara Emery', 
              'Grant Fairchild', 
              'Yi Gao', 
              'Michael Gomez', 
              'Matt Harrison', 
              'Desiree Holler', 'Ivana Ilic', 'Jung Min Kim', 'Taissa Lytchenko', 'Stephanie Otto', 'Carissa Romero', 
              'Alexandra Scurry', 
              'Joe Z']

idx = np.random.permutation(15)

for ii in idx:
    print(class_list[ii])



