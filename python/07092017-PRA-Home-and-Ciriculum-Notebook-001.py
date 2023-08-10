import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
#import matplotlib.animation as animation

get_ipython().magic('matplotlib inline')

weight =[258.1,257.1,256.6,257.7,257.6,254.3,252.5,252.6,251.7,251.2,250.1,247.8] 

#plot(weight, 'm', label='line1', linewidth=4)
plt.title('Q2 2017 - Progress on Weight Loss Program')
plt.grid(True)
plt.xlabel('Weigh in #')
plt.ylabel('Weight in Lbs.')
ax = plt.gca()
at = AnchoredText(
        "Rob's Weekly Weight Loss progress",
        loc=3, prop=dict(size=10), frameon=True,
    )
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

plt.plot(weight,'m', linewidth=4, linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)

import pandas as pd
import numpy as np

df = pd.DataFrame(data=np.array([[1,2,3,4], [5,6,7,8]], dtype=int), columns=['Pacific','Mountain','Central','Eastern'])
plt.plot(df)
df

from IPython.display import display, Math, Latex
display(Math(r'\sqrt{a^2 + b^3}'))

#%lsmagic

#%quickref

#%debug

