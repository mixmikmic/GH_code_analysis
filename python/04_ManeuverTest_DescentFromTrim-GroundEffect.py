# Notebook Initialization
get_ipython().run_line_magic('run', './python/nb_init.py')
get_ipython().run_line_magic('matplotlib', 'inline')

import plotting_routines as plrt
import data_handling as dh

HTML(open('./style/style_unina_iwes.css', 'r').read())

# Directory for output
dest_folder='../output/mantest_descent0_ap/'

# Catalog to file
os.system('../JSBSim/JSBSim --root=../JSBSim/ --catalog=c172x_unina > ../JSBSim/catalog_c172x_unina.txt')

# Reference to script file
script_path='../JSBSim/scripts/c172x_mantest_descent0_ap.xml'

# Reference to initialization file
init_path = '../JSBSim/aircraft/c172x_unina/init_mantest_descent0_ap.xml'

dh.show_file(init_path)

dh.show_file(script_path)

# JSBSim script launch command
os.system('../JSBSim/JSBSim --root=../JSBSim/ --script='+script_path+' > ../JSBSim/log_descent0_ap.txt')

dh.move_files_to_folder('*.csv',dest_folder,'../JSBSim/')
dh.move_files_to_folder('*.txt',dest_folder,'../JSBSim/')

# Data extraction from JSBSim custom output files
data_fcs = np.genfromtxt(dest_folder+'C172x_unina_fcs.csv',       delimiter=',', skip_header=1)
data_vel = np.genfromtxt(dest_folder+'C172x_unina_velocities.csv',delimiter=',', skip_header=1)
data_att = np.genfromtxt(dest_folder+'C172x_unina_attitude.csv',  delimiter=',', skip_header=1)
data_aer = np.genfromtxt(dest_folder+'C172x_unina_aero.csv',      delimiter=',', skip_header=1)
data_pos = np.genfromtxt(dest_folder+'C172x_unina_position.csv',  delimiter=',', skip_header=1)
data_eng = np.genfromtxt(dest_folder+'C172x_unina_propulsion.csv',delimiter=',', skip_header=1)

# Time histories
plrt.plot_Cmd_AngVel_EulerAng(data_fcs, data_vel, data_att, dest_folder)
plrt.plot_Alfa_Beta_V(data_aer, data_vel, dest_folder)
plrt.plot_PosGeoc(data_pos, data_vel, dest_folder)
#plrt.plot_EngineStatus(data_eng, data_vel, dest_folder)

## Ground effect: a closer look
plrt.plot_Ground_Effect(data_aer, dest_folder)

# Plotting in NEA reference frame
import geography as geo
import plotting_utilities as plut

t_pos     = (data_pos[:,0]*unit.s).magnitude
h_sl      = ((data_pos[:,1]*unit.ft).to(unit.m)).magnitude
lat_deg   = (data_pos[:,7]*unit.deg).magnitude
lon_deg   = (data_pos[:,9]*unit.deg).magnitude
lat_rad   = ((lat_deg*unit.deg).to(unit.rad)).magnitude
lon_rad   = ((lon_deg*unit.deg).to(unit.rad)).magnitude

r = geo.geoc_to_NEA(lat_rad,lon_rad,h_sl,flat=1)

plrt.plot_traj2D_NEA(r,t_pos,n_arrows=8,arrow_size=2, dest_folder=dest_folder)
plrt.plot_traj3D_NEA(r,'S','W',view=(45,45),to_scale='XYZ',mrk_size=80,dest_folder=dest_folder)



