# import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Bio.Data.CodonTable
from Bio.SubsMat import MatrixInfo as submats
import networkx as nx
from src.codonUtils import utils
from src.codonOptimizer import tableOptimizer
from src.codonTable import codonTable
import bct
import random
import pickle

help(utils)

help(codonTable)

# Demonstrate CodonTable visualization
get_ipython().magic('matplotlib inline')
test = codonTable()
fig = test.plot3d('Standard Codon Table: Node Color=Hydropathy')
fig2 = test.plotGraph('Standard Codon Table: Node Color=Residue Degeneracy', nodeSize='count', nodeColor='kd')

help(tableOptimizer)

# Demonstrate MonteCarlo simulation
sim = tableOptimizer()
optimizedTable, Ws, Es = sim.GDA()
optimizedTable = codonTable(optimizedTable)
standardTable = codonTable()

# Compare tables
fig1 = standardTable.plot3d('Standard Code: Node Color=Hydropathy')
fig2 = standardTable.plotGraph('Standard Code: Node Color=Residue Degeneracy', nodeColor='kd', nodeSize='count')
fig3 = optimizedTable.plot3d('Optimized Table: Node Color=Hydropathy')
fig4 = optimizedTable.plotGraph('Optimzied Table: Node Color=Residue Degeneracy', nodeColor='kd', nodeSize='count')

