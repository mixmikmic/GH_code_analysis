get_ipython().run_line_magic('run', './python/nb_init.py')
get_ipython().run_line_magic('matplotlib', 'inline')

import data_handling as dh
from ACDataPlots import ACDataPlots

HTML(open('./style/style_unina_iwes.css', 'r').read())

# Directory for output
dest_folder='../output/mantest_glide_hold_ap/'

# Catalog to file
os.system('../JSBSim/JSBSim --root=../JSBSim/ --catalog=c172x_unina > ../JSBSim/catalog_c172x_unina.txt')

# Reference to script file
script_path='../JSBSim/scripts/c172x_mantest_glide_hold_ap.xml'

# Reference to initialization file
init_path = '../JSBSim/aircraft/c172x_unina/init_mantest_glide_hold_ap.xml'

dh.show_file(script_path)

# JSBSim script launch command
os.system('../JSBSim/JSBSim --root=../JSBSim/ --script='+script_path+' > ../JSBSim/log_mantest_glide_hold_ap.txt')

dh.move_files_to_folder('*.csv',dest_folder,'../JSBSim/')
dh.move_files_to_folder('*.txt',dest_folder,'../JSBSim/')

# Extract data arrays from output .csv files
sim = ACDataPlots('C172x_unina',dest_folder)

# Plot time histories
sim.plot_Commands()
sim.plot_gamma()
sim.plot_AltitudeVelocity()
sim.plot_EulerAng()

sim.get_traj_in_NEA()
sim.plot_GroundTrack()
sim.plot_traj3D_in_NEA(view=(45,45),Y_proj='W')

