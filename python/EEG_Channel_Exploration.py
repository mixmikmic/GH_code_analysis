import scipy.io
import numpy as np

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

#Replace with your relevant path
data = scipy.io.loadmat("../BCI_Comp_III_Wads_2004/Subject_A_Train.mat")



print(data['Flashing'].shape)
print(data['Signal'].shape)

data['Signal'][0]

data['Signal'][0][0]

np.sum(data['Flashing'] == 1)/(85*7794)

#240 Hz
print(7794*1/240, 'total seconds per trial')

print('Roughly', 12*.175 * 240, ' sized chunks per one sequence of twelve flashes')

print('Roughly this many sequences per trial:',7794/504)

array = []

for i in range(16):
    chunk = 504
    array.append(data['Flashing'][0][504*i:504*(i+1)])

array[-1].shape #size of remainder piece

start = 0
end = 7794
trial = 0

channels = [8, 10, 12, 48, 50, 52, 60, 62] #true relevant channels

data['Flashing'][0].size

character = []
character_signals = []
character_labels = []
epochs = 15
flashes = 12

for flash in range(epochs*flashes):
    # for the signals
    one_flash = []
    for chanel in channels:
        chanel_signals = []
        for signal in range(96):
            chanel_signals.append(float(data['Signal'][0][signal][chanel]))
        print(chanel_signals)
        one_flash.append(chanel_signals)
    print(one_flash)
    character_signals.append(one_flash)
    # for the labels
    label_index = flash * 42
    character_labels.append(start)

character.append(character_signals)
character.append(character_labels)
np_character = np.array(character)
np_character.shape

labels = []
for i in range(180):
    labels.append(data['StimulusType'][0][i*42])

np_character_signals = np.array(character_signals)
np_character_signals.shape

