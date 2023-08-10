# Imports
import random
import pickle
import numpy as np
np.random.seed(seed=0)
get_ipython().run_line_magic('matplotlib', 'inline')

# Import: Our code
import nb_utils
import GP_deriv
import GP_vec
from MAS_exp import ContinuousEnvironment
from utils_synthetic import plot_laplacian

load_dict = True 

N = 200 # Time-resolution
M = 10  # Space-resolution
agent_positions = [[2.,1.], [2,-1], [-2,-1], [-2,1]]

env = ContinuousEnvironment(N, M)
env.initialise_agents(agent_positions)

# means:
mu1 = lambda x: np.array([[-1.5], [0.]])
mu2 = lambda x: np.array([[1.5], [0.]])
mean_list = list([mu1,mu2])
# covariances
cov1 = lambda x: (np.sin(x) + 2.1) * np.eye(2)
cov2 = lambda x: (np.cos(x) + 2.1) * np.eye(2)
cov_list = list([cov1,cov2])

env.calculate_trajectories(mean_list, cov_list)
x_train, y_train, t_train, x_test, y_test, t_test = env.get_trajectories()

env.potential(mean_list,cov_list)

if not load_dict:
    traj_model = GP_vec.Multiple_traj(x_train, y_train, t_train, x_test, y_test, t_test)    
    
    dic = {'X_true':env.X_test, 'Y_true':env.Y_test,
           'lap_true': env.div_frames, 'lap_inf': traj_model.div_mesh,
           'xloc_syn': traj_model.loc_time_ind_mesh_x, 'yloc_syn': traj_model.loc_time_ind_mesh_y,
           'skl_div_syn': traj_model.KL_div_mesh_signed
            }
    
    del traj_model # Free the memory for our trajectory model.        
    # pickle.dump( dic, open( "./saved_models/exp_synthetic_varyingstrength.p", "wb" ) ) # Uncomment for saving new model    
else:
    dic = pickle.load( open( "./../saved_models/exp_synthetic_varyingstrength.p", "rb" ) )

plot_laplacian(dic['X_true'], dic['Y_true'], 
               dic['lap_true'], dic['lap_inf'], 
               dic['xloc_syn'], dic['yloc_syn'], 
               dic['skl_div_syn'], T=[56,77,97,113,146], state = 'var')

