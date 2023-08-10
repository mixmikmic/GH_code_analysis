import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

abalone_data = np.genfromtxt('./Datasets/Dataset.data', delimiter=' ') 
abalone_data = np.delete(abalone_data, 0, 1)
# We are skiping the sex for we do not know what to do with that kind of data

NbAttributes = np.shape(abalone_data)[1] #8
NbCases = np.shape(abalone_data)[0]      #4177
lables = ['length', 'diameter','height','whole_weight','Shucked_weight','viscera_weight','shell_weight', 'rings']

t = PrettyTable(lables)
for i in range(NbCases):
    t.add_row( [abalone_data[i][j] for j in range(NbAttributes)] )

print(t.get_string(start = 0, end = 10))

#abalone_data = abalone_data.T
rings = abalone_data[7]

for i in range(NbAttributes - 1):
    plt.figure(figsize=(12, 3), dpi=80,)
    plt.plot(abalone_data[i], rings, 'ro')
    plt.title(lables[i])

plt.show()



