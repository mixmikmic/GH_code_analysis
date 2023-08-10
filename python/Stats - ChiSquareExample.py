from IPython.core.display import Image, display
import numpy as np
from scipy.stats import chisquare

display(Image(url='ChiSquare.png', width=150, height=150))

chisquare([16, 18, 16, 14, 12, 12])

chisquare([50, 35, 16, 14, 12, 5])

chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8])

chisquare([10, 10, 10, 10, 10, 5], f_exp=[50, 50, 50, 50, 50, 25])

from scipy.stats import chi2_contingency
obs = np.array([[43,9], [44,4]])
chi2_contingency(obs)

obs = np.array([[43,9], [44,4]]).T
obs.shape

chisquare(obs)
chisquare(obs, axis=None)

