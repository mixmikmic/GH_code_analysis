#NAME: Dynamic Ising Model
#DESCRIPTION: Glauber dynamics of the Ising model.

import numpy as np
from numpy import random
from scipy.misc import factorial

from IPython.display import display
from ipywidgets import widgets, interact
from bokeh.plotting import figure, show, output_notebook
from bokeh.io import push_notebook, gridplot

#prepare notebook for inline plots
output_notebook()
#seed random number generator for spin-flips
random.seed()

#set lattice parameters
grid_size = 20
N = grid_size**2
#set interaction properties
interaction_range = 1.0
#set boundary conditions as "hard wall" or "cyclic"
boundary_conditions = "hard wall"

if boundary_conditions != "hard wall" and boundary_conditions != "cyclic":
    print("invalid boundary conditions. Choose 'hard wall' or cyclic'")

#each spin-1/2 lattice site is an instance of the element class
class element:
    def __init__(self, spin): #spin will take vals of 0 or 1
        self.spin = spin    #representing down and up
    def flip(self, temperature, up_energy, down_energy): #temperature between 0 and 1
        up_boltzmann = np.exp(-up_energy/temperature)
        down_boltzmann = np.exp(-down_energy/temperature)
        prob_up = up_boltzmann/(up_boltzmann + down_boltzmann)
        if random.uniform(0.0,1.0) < prob_up:
            self.spin = 1
        else:
            self.spin = 0

def initialise_lattice():
    lattice = []
    #initialise lattice as all spin-down
    for i in range(grid_size):
        lattice.append([])
        for j in range(grid_size):
            lattice[-1].append(element(0))
    #convert to numpy array for iteration
    lattice = np.asarray(lattice)
    return lattice

def get_spins(lattice):
    x, y, spins = [], [], []
    it = np.nditer(lattice, ['multi_index', 'refs_ok'])
    for element in it:
        ind_x, ind_y = it.multi_index
        x.append(ind_x)
        y.append(ind_y)
        spins.append(lattice[ind_x][ind_y].spin)
    return x, y, spins

def calculate_energy(lattice, index, exchange_energy, interaction_range): #i and j are indices of the element
    i,j = index
    neighbour_indices = [] 
    #find spins which interact with the one at i,j
    it = np.nditer(lattice, ['multi_index', 'refs_ok'])
    if boundary_conditions == "cyclic":
        for element in it:
            k, l = it.multi_index
            #cyclic boundary conditions imposed
            if (((i-k)%grid_size)**2 + ((j-l)%grid_size)**2) <= interaction_range**2:
                neighbour_indices.append([k, l])  
    elif boundary_conditions == "hard wall":
        for element in it:
            k, l = it.multi_index
            #cyclic boundary conditions imposed
            if ((i-k)**2 + (j-l)**2) <= interaction_range**2:
                neighbour_indices.append([k, l])             
    #find number of up neighbours
    up_neighbours = 0 
    for k in neighbour_indices:
        if lattice[k[0]][k[1]].spin == 1:
            up_neighbours += 1 
    #calculate energy that the spin would have if up or if down
    up_energy = exchange_energy*up_neighbours
    #down neighbours = 4 - up neighbours
    down_energy = exchange_energy*(4 - up_neighbours)
    return up_energy, down_energy

def spin_flip(lattice, temperature, exchange_energy, interaction_range):
    #create iterable object
    it = np.nditer(lattice, ['multi_index', 'refs_ok'])
    for element in it:
        up_energy, down_energy = calculate_energy(lattice, it.multi_index, exchange_energy, interaction_range)
        lattice[it.multi_index].flip(temperature, up_energy, down_energy)
        
#calculate the entropy of the system
#k_B = 1
def entropy(spins):
    N_down = spins.count(0)
    if N_down == 0:
        N_down = 1 #to avoid log(0) errors    
    N_up = spins.count(1)
    if N_up == 0:
        N_up = 1     
    #use Stirling's approximation to find entropy
    S = (N_up+N_down)*np.log(N_up+N_down) - N_down*np.log(N_down) - N_up*np.log(N_up)
    #return entropy of current system state
    return S
        
def create_plot_data(lattice):
    x, y, spins = get_spins(lattice)
    #red for spin up, blue for spin down
    colors = ["#%02x%02x%02x" % (0, int(200*g), int(200*b)) for g, b in zip(spins, spins)]
    return x,y,colors,spins

lattice = initialise_lattice()
#start at high temperature with no exchange interaction...
spin_flip(lattice, 10.0, 0.0, 1.0)

x,y,colors,spins = create_plot_data(lattice)
p1 = figure(height = 500, width = 500)
q1 = p1.square(x, y, fill_color = colors, line_color = None, size = 20)

entropies = []
steps = []
p2 = figure(height = 500, width = 500)
q2 = p2.line(steps, entropies)
p2.xaxis.axis_label = "Step"
p2.yaxis.axis_label = "Entropy"

p = gridplot([[p1, p2]])

#create a temperature slider
T_slider = widgets.FloatSlider(value = 10.0, min = 0.001, max = 10.0, description = "Temperature")

#create an exchange energy slider
EE_slider = widgets.FloatSlider(value = 0.0, min = -10.0, max = 10.0, description = "Exchange Energy")

#create an interaction range slider
range_slider = widgets.FloatSlider(value = 1.0, min = 1.0, max = 10.0, description = "Interaction Range")

#create buttons to implement these changes
single_update_button = widgets.Button(description="Update")
multi_update_button = widgets.Button(description="Multi-update")

#temperature and exchange energy buttons
def update(b):
    
    #update grid
    spin_flip(lattice, T_slider.value, EE_slider.value, range_slider.value)
    q1.data_source.data['fill_color'] = create_plot_data(lattice)[2]
    
    #update global variables 'step' and 'entropies'
    entropies.append(entropy(create_plot_data(lattice)[3])) 
    if len(steps) == 0:
        steps.append(1)
    else:
        steps.append(steps[-1]+1)
    #update entropy plot
    q2.data_source.data['entropies'] = entropies
    q2.data_source.data['steps'] = steps
    push_notebook()
    
def multi_update(b):
    #run update 20 times
    for i in range(20):
        update(b)
    
#update the plots when the button is clicked
single_update_button.on_click(update)
multi_update_button.on_click(multi_update)

show(p) #show the plots

#UI for energetics
display(T_slider, EE_slider, range_slider, single_update_button, multi_update_button)

