import numpy as np
from MDPlus.core import Fasu, Cofasu
from MDPlus.analysis import pca
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'font.size': 15}) #This sets a better default label size for plots
# Load the data for the wt and irhy simulations into two "Fasu"s:
f_wt = Fasu('wt_ca.binpos', top='wt_ca.pdb')
f_irhy = Fasu('irhy_ca.binpos', top='irhy_ca.pdb')
#Combine the two sets of trajectory data into one "Cofasu":
trajdata = Cofasu([f_wt, f_irhy])
# Create labels for the two datasets:
datanames = ['wt', 'irhy']

# define the plotting function:
def plot_rmsd(cofasu, datanames):
    """
    This function takes a cofasu and a list of data names and produces an rmsd plot.
    
    """
    cofasu.align() # least squares fits each snapshot to the first.
    frames_per_set = len(cofasu) // len(datanames) # we assume each trajectory file is the same length.
    for i in range(len(datanames)):
        # The next two lines do the rmsd calculation:
        diff = cofasu[i * frames_per_set : (i + 1) * frames_per_set] - cofasu[0]
        rmsd = np.sqrt((diff * diff).sum(axis=2).mean(axis=1))
        plt.plot(rmsd, label=datanames[i]) # plot the line for this dataset on the graph.
    plt.xlabel('Frame number')
    plt.ylabel('RMSD (Ang.)')
    plt.legend(loc='lower right')

# now use the plotting function:    
plot_rmsd(trajdata, datanames)

def plot_rmsf(cofasu):
    """
    Plots the root mean square fluctuations of the atoms in a cofasu.
    
    """
    diff = cofasu[:] - cofasu[:].mean(axis=0)
    rmsf = np.sqrt((diff * diff).sum(axis=2).mean(axis=0))
    plt.xlabel('atom number')
    plt.ylabel('RMSF (Ang.)')
    plt.plot(rmsf)
    
plt.figure(figsize=(15,5))
plt.subplot(121)
frames_per_set = len(trajdata) // len(datanames)
plot_rmsf(trajdata[:frames_per_set]) # the first half of the cofasu has the wt data.
plt.title(datanames[0])
plt.subplot(122)
plot_rmsf(trajdata[frames_per_set:]) # the second half of the cofasu has the irhy data.
plt.title(datanames[1])

# Reload the selected portion of the data for the wt and irhy simulations:
f_wt = Fasu('wt_ca.binpos', top='wt_ca.pdb', selection='resid 1 to 370')
f_irhy = Fasu('irhy_ca.binpos', top='irhy_ca.pdb', selection='resid 1 to 370')
trajdata = Cofasu([f_wt, f_irhy])

plot_rmsd(trajdata, datanames)

# First define a plotting function:
def plot_pca(pca_model, datanames, highlight=None):
    """
    Plots the projection of each trajectory in the cofasu in the PC1/PC2 subspace.
    
    If highlight is a number, this dataset is plotted in red against all others in grey.
    
    """
    p1 = pca_model.projs[0] # the projection of each snapshot along the first principal component
    p2 = pca_model.projs[1] # the projec tion along the second.

    frames_per_rep = len(p1) // len(datanames) # number of frames (snapshots) in each dataset - assumed equal length
    for i in range(len(datanames)):
        start = i * frames_per_rep
        end = (i + 1) * frames_per_rep
        if highlight is None: # each dataset is plotted with a different colour
            plt.plot(p1[start:end], p2[start:end], label=datanames[i]) 
            plt.text(p1[start], p2[start], 'start')
            plt.text(p1[end-1], p2[end-1], 'end')
        else:
            if i != highlight:
                plt.plot(p1[start:end], p2[start:end], color='grey')
    if highlight is not None:
        start = highlight * frames_per_rep
        end = (highlight + 1) * frames_per_rep
        plt.plot(p1[start:end], p2[start:end], color='red', label=datanames[highlight])
        plt.text(p1[start], p2[start], 'start')
        plt.text(p1[end-1], p2[end-1], 'end')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper left')

# Now use it:
p = pca.fromtrajectory(trajdata)
plot_pca(p, datanames)



