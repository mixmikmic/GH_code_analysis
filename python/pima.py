get_ipython().magic('matplotlib inline')

import numpy as np
import seaborn as sns
sns.axes_style('whitegrid')

from sklearn.ensemble import RandomForestClassifier
from pima import Pima

seed = np.random.randint(2**16)
print "seed: {}".format(seed)

estimator = RandomForestClassifier(max_features=0.3, min_samples_split=2, n_estimators=32, 
                                   max_depth=3, min_samples_leaf=1, random_state=seed)

pima = Pima(estimator)
pima.train()
pima.experience_curve().show()
pima.plot_roc_curve().show()



