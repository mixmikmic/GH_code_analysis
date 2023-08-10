# Import Modules
import numpy as np

# Create a list
regimentSize = [534, 5468, 6546, 542, 9856, 4125]

# Create a ndarray from the regimentSize list
regimentSizeArray = np.array(regimentSize) ;regimentSizeArray

# What are the number of dimensions of the array?
regimentSizeArray.ndim

# What is the shape of the array?
regimentSizeArray.shape

# Create two lists
regimentSizePreWar = [534, 5468, 6546, 542, 9856, 4125]
regimentSizePostWar = [234, 255, 267, 732, 235, 723]


# Create a ndarray from a nested list
regimentSizePerPostArray = np.array([regimentSizePreWar,regimentSizePostWar]); regimentSizePerPostArray

# What are the number of dimensions of the array?
regimentSizePerPostArray.ndim

# What is the shape of the array?
regimentSizePerPostArray.shape

