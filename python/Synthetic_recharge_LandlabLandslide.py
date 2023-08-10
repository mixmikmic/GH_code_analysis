#import Python utilities for calculating and plotting
import six
import os
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np

#import utilities for importing and exporting to HydroShare
from utilities import hydroshare
# set variables for interacting with HydroShare from this notebook
hs=hydroshare.hydroshare()
# Create object to map the home directory
homedir = r'/home/jovyan/work/notebooks/data/' + str(os.environ["HS_RES_ID"]) + '/' + str(os.environ["HS_RES_ID"]) + '/data/contents/'
print homedir

# Import Landlab libraries
import landslide_probability
from landslide_probability import LandslideProbability
from landlab import RasterModelGrid
from landlab import imshow_grid_at_node

grid = RasterModelGrid((5, 4), spacing=(0.2, 0.2))

grid.number_of_nodes

grid.nodes

core_nodes = grid.core_nodes
core_nodes

sorted(LandslideProbability.input_var_names)

LandslideProbability._var_doc

LandslideProbability._var_units

gridnodes = grid.number_of_nodes
grid_size = grid.number_of_nodes

grid['node']['soil__density']=          2000. * np.ones(gridnodes)

grid['node']['soil__internal_friction_angle']=          np.sort(np.random.randint(26, 37, gridnodes))

grid['node']['soil__mode_total_cohesion']=          np.sort(np.random.randint(30, 900, gridnodes))
    
scatter_dat = np.random.randint(1, 10, gridnodes)    
grid['node']['soil__maximum_total_cohesion']=          grid.at_node['soil__mode_total_cohesion'] + scatter_dat

grid['node']['soil__minimum_total_cohesion']=          grid.at_node['soil__mode_total_cohesion'] - scatter_dat

grid['node']['soil__thickness']=          np.sort(np.random.randint(1, 10, gridnodes))

grid['node']['soil__transmissivity']=          np.sort(np.random.randint(5, 20, gridnodes),-1)
        
grid['node']['topographic__slope'] = np.random.rand(gridnodes)

grid['node']['topographic__specific_contributing_area']=          np.sort(np.random.randint(30, 900, gridnodes))

plt.figure('Slope')
imshow_grid_at_node(grid,'topographic__slope', cmap='copper_r',
                 grid_units=('coordinates', 'coordinates'), shrink=0.75,
                 var_name='Slope', var_units='m/m')
plt.savefig('Slope.png')

n = 25

distribution1 = 'uniform'
Remin_value = 5 
Remax_value = 15 

LS_prob1 = LandslideProbability(grid,number_of_iterations=n,
    groudwater__recharge_distribution=distribution1,
    groundwater__recharge_min_value=Remin_value,
    groundwater__recharge_max_value=Remax_value)
print('Distribution = '), LS_prob1.groundwater__recharge_distribution
print('Uniform recharge successfully instantiated')

distribution2 = 'lognormal'
Remean = 5.
Restandard_deviation = 0.25
LS_prob2 = LandslideProbability(grid,number_of_iterations=n,
    groundwater__recharge_distribution=distribution2,
    groundwater__recharge_mean=Remean,
    groundwater__recharge_standard_deviation=Restandard_deviation)
print('Distribution = '), LS_prob2.groundwater__recharge_distribution
print('Lognormal recharge successfully instantiated')

distribution3 = 'lognormal_spatial'
Remean3 = np.random.randint(2,7,grid_size)
Restandard_deviation3 = np.random.rand(grid_size)
LS_prob3 = LandslideProbability(grid,number_of_iterations=n,
    groundwater__recharge_distribution=distribution3,
    groundwater__recharge_mean=Remean3,
    groundwater__recharge_standard_deviation=Restandard_deviation3)
print('Distribution = '), LS_prob3.groundwater__recharge_distribution
print('Lognormal spatial recharge successfully instantiated')

HSD_dict = {}
for vkey in range(2,8):
    HSD_dict[vkey] = np.random.randint(20,120,10)
print('HSD_dict dictionary is a unique array of recharge provided as arrays (‘values’) for each of the Hydrologic Source Domain (HSD) (‘keys’). ')    
print('The first key of this dictionary is:')
HSD_dict.keys()[0]

HSD_id_dict = {}
for ckey in grid.core_nodes:
    HSD_id_dict[ckey] = np.random.randint(2,8,2)
print('The first key:value pair of this dictionary is:')
HSD_id_dict.items()[0]

fract_dict = {}
for ckey in grid.core_nodes:
    fract_dict[ckey] =  np.random.rand(2)
print('The fractions (values) assigned to the first node (key) are: ')
fract_dict.values()[0]

distribution4 = 'data_driven_spatial'
HSD_inputs = [HSD_dict,HSD_id_dict,fract_dict]
LS_prob4 = LandslideProbability(grid,number_of_iterations=n,
    groundwater__recharge_distribution=distribution4,
    groundwater__recharge_HSD_inputs=HSD_inputs)
print('Distribution = '), LS_prob4.groundwater__recharge_distribution
print('Data driven spatial recharge successfully instantiated')

LS_prob1.calculate_landslide_probability()
print('Landslide probability successfully calculated')

sorted(LS_prob1.output_var_names)

LS_prob1_probability_of_failure = grid.at_node['landslide__probability_of_failure']
grid.at_node['landslide__probability_of_failure']

LS_prob1_relative_wetness = grid.at_node['soil__mean_relative_wetness']
grid.at_node['soil__mean_relative_wetness']

LS_prob2.calculate_landslide_probability()
LS_prob2_probability_of_failure = grid.at_node['landslide__probability_of_failure']
LS_prob2_relative_wetness = grid.at_node['soil__mean_relative_wetness']

LS_prob3.calculate_landslide_probability()
LS_prob3_probability_of_failure = grid.at_node['landslide__probability_of_failure']
LS_prob3_relative_wetness = grid.at_node['soil__mean_relative_wetness']

LS_prob4.calculate_landslide_probability()
LS_prob4_probability_of_failure = grid.at_node['landslide__probability_of_failure']
LS_prob4_relative_wetness = grid.at_node['soil__mean_relative_wetness']

fig = plt.figure('Probability of Failure')
xticks = np.arange(-0.1, 0.8, 0.4)
ax1 = fig.add_subplot(221)
ax1.xaxis.set_visible(False)
imshow_grid_at_node(grid, LS_prob1_probability_of_failure, plot_name='Recharge 1',
                    allow_colorbar=False, cmap='OrRd',
                    grid_units=('coordinates',''))
ax2 = fig.add_subplot(222)
ax2.xaxis.set_visible(False)
imshow_grid_at_node(grid, LS_prob2_probability_of_failure, plot_name='Recharge 2',
                    allow_colorbar=False, cmap='OrRd',
                    grid_units=('coordinates', 'coordinates'))
ax3 = fig.add_subplot(223)
ax3.set_xticks(xticks)
imshow_grid_at_node(grid, LS_prob3_probability_of_failure,plot_name='Recharge 3',
                    allow_colorbar=False, cmap='OrRd',
                    grid_units=('coordinates', 'coordinates'))
ax4 = fig.add_subplot(224)
ax4.set_xticks(xticks)
imshow_grid_at_node(grid, LS_prob4_probability_of_failure, cmap='OrRd', plot_name='Recharge 4',
                    grid_units=('coordinates', 'coordinates'), shrink=0.9,
                    var_name='Probability of Failure')
plt.savefig('Probability_of_Failure_synthetic.png')

fig = plt.figure('Mean Relative Wetness')
xticks = np.arange(-0.1, 0.8, 0.4)
ax1 = fig.add_subplot(221)
ax1.xaxis.set_visible(False)
imshow_grid_at_node(grid, LS_prob1_relative_wetness, plot_name='Re Opt. 1',
                    allow_colorbar=False, cmap='YlGnBu',
                    grid_units=('coordinates',''))
ax2 = fig.add_subplot(222)
ax2.xaxis.set_visible(False)
imshow_grid_at_node(grid, LS_prob2_relative_wetness, plot_name='Re Opt. 2',
                    allow_colorbar=False, cmap='YlGnBu',
                    grid_units=('coordinates', 'coordinates'))
ax3 = fig.add_subplot(223)
ax3.set_xticks(xticks)
imshow_grid_at_node(grid, LS_prob3_relative_wetness,plot_name='Re Opt. 3',
                    allow_colorbar=False, cmap='YlGnBu',
                    grid_units=('coordinates', 'coordinates'))
ax4 = fig.add_subplot(224)
ax4.set_xticks(xticks)
imshow_grid_at_node(grid, LS_prob4_relative_wetness, cmap='YlGnBu', plot_name='Re Opt. 4',
                    grid_units=('coordinates', 'coordinates'), shrink=0.9,
                    var_name='Mean Relative Wetness')
plt.savefig('Mean_Relative_Wetness_synthetic.png')



