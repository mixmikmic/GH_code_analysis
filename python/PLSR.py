import pandas as pd
import numpy as np
import hoggorm as hogg
import hoggormplot as hplot
import matplotlib.pyplot as plt

fluorescence = np.array(pd.read_table('https://raw.githubusercontent.com/khliland/hoggormExamples/master/data/cheese_fluorescence.txt',                              header = None, sep = '\s+'))
sensory = np.array(pd.read_table('https://raw.githubusercontent.com/khliland/hoggormExamples/master/data/cheese_sensory.txt',                              header = None, sep = '\s+'))

SENSORYplsr = hogg.nipalsPLS2(fluorescence, sensory, 4, cvType = ["loo"])

SENSORYplsr.Y_cumCalExplVar()

SENSORYplsr.Y_RMSECV()

hplot.explainedVariance(SENSORYplsr)

hplot.explainedVariance(SENSORYplsr, individual = True)

hplot.scores(SENSORYplsr)

hplot.correlationLoadings(SENSORYplsr)

NIR = np.array(pd.read_table('https://raw.githubusercontent.com/khliland/hoggormExamples/master/data/gasoline_NIR.txt',                              header = None, sep = '\s+'))
octane = np.array(pd.read_table('https://raw.githubusercontent.com/khliland/hoggormExamples/master/data/gasoline_octane.txt',                              header = None, sep = '\s+'))

NIRplsr = hogg.nipalsPLS1(NIR, octane, 10, cvType = ["loo"])

NIRplsr.Y_cumCalExplVar()

NIRplsr.Y_RMSECV()

hplot.explainedVariance(NIRplsr)

hplot.scores(NIRplsr)

hplot.loadings(NIRplsr, weights=True, line=True)

hplot.predict(NIRplsr, comp=3)

hplot.coefficients(NIRplsr, comp=3)

scores = NIRplsr.X_scores()[:,:2]
plt.scatter(scores[:,0],scores[:,1], c = octane)
plt.colorbar()
plt.show()

airpolutionPD = pd.read_csv('https://raw.githubusercontent.com/khliland/hoggormExamples/master/data/airpolution.csv', index_col=0)
airpolution = np.array(airpolutionPD)

AIRPOLUTIONplsr = hogg.nipalsPLS2(airpolution[:,:3], airpolution[:,3:], 3, cvType = ["loo"])

hplot.explainedVariance(AIRPOLUTIONplsr, individual=True)

hplot.biplot(AIRPOLUTIONplsr, XvarNames=list(airpolutionPD.columns.values[:3]))

hplot.biplot(AIRPOLUTIONplsr, which="Y", YvarNames=list(airpolutionPD.columns.values[3:]))

