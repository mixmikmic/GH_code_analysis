import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
sns.set_style("white")

import time
import timeit

import scipy.stats 
import pandas as pd
import pymc as pm
import pickle

# Import data
game16_df = pd.read_csv('clean-data/game_outcomes_15-16.csv')
player_info = pd.read_csv('clean-data/player_info_pergame.csv')

score_diff = game16_df['diff'].values
#plot a histogram
plt.hist(score_diff,color='y',alpha = 0.5)
plt.title("Distribution of score difference in year 2016")
plt.xlabel('The Score difference')
plt.ylabel('Frequency')

# Here we have sigma for the entire data set and assume this sigma as my likehood variance as well
print "The variance of the entire data set is:",np.var(score_diff)

#plot a histogram
ortg_values = player_info['PTS'].values
plt.hist(ortg_values,color='y',bins=15,alpha =0.5)
plt.title("Distribution of offensive Rating in year 2016")
plt.xlabel('The Offensive Rating difference')
plt.ylabel('Frequency')

# import the pickle files for guest/host teams line_up
with open('clean-data/guest_team_line_up.pkl', 'rb') as handle:
    guest_team_line_up = pickle.load(handle)
with open('clean-data/host_team_line_up.pkl','rb') as handle:
    host_team_line_up = pickle.load(handle)

def make_guest_host_mat(game_outcomes_df):
    '''
    Makes a matrix for guests and hosts. Each row of each
    matrix corresponds to one matchup. All elements of each row
    are zero except the ith one (different for each row).
    For the guest matrix, the ith entry in row j means that in game j,
    the guest team was team i. In the host matrix, the ith entry in
    row j means that the host team was team i
    '''
    
    def make_matrix(mat, indices):
        '''given a matrix and indices, sets the right one in each row
        to be true'''
        for (i, ind) in enumerate(indices):
            mat[i, ind] = True
        
    nrows = game_outcomes_df.shape[0]
    ncols = np.max(game_outcomes_df['Visitor_Index'] + 1)
    
    guest_matrix = np.zeros((nrows, ncols), dtype = bool)
    host_matrix = np.zeros((nrows, ncols), dtype = bool)
    
    make_matrix(guest_matrix, game_outcomes_df['Visitor_Index'].values)
    make_matrix(host_matrix, game_outcomes_df['Home_Index'].values)
    
    return(guest_matrix, host_matrix)

guest_matrix, host_matrix = make_guest_host_mat(game16_df)

# Construct beta and gamma 
team_num = 30
player_num = len(player_info)

coefs = pm.MvNormalCov("coefs", mu = np.zeros(2*(team_num + player_num)),
                      C = np.eye(2*(team_num + player_num)))

# beta0 = np.empty(team_num,dtype=object)
# beta0 = pm.Container(np.array([pm.Normal('beta0_%i' % i, 
#                    mu=1, tau=1) for i in xrange(team_num)]))
# gamma0 = np.empty(team_num,dtype=object)
# gamma0 = pm.Container(np.array([pm.Normal('gamma0_%i' % i, 
#                    mu=1, tau=1) for i in xrange(team_num)]))
# betas = pm.Container(np.array([pm.Normal('betas_%i' % i, 
#                    mu=1, tau=1) for i in xrange(play_num)]))
# gammas = pm.Container(np.array([pm.Normal('gammas_%i' % i, 
#                    mu=1, tau=1) for i in xrange(play_num)]))
tau = pm.Gamma("tau", alpha = 2, beta = 2)

# Copy the variables from Andy's part
guest_lineup_arr = np.array(guest_team_line_up)
host_lineup_arr = np.array(host_team_line_up)
off_rating = player_info['PTS'].values + player_info['AST'].values
def_rating = player_info['BLK'].values + player_info["STL"].values + player_info['DRB'].values

def split_params(coefs, nplayers, nteams):
    '''
    Split the parameters
    first are the beta0 for each team
    then the beta for each player
    then the gamma0 for each team
    then the gamma for each player'''
    assert(coefs.shape == (2*(nplayers+nteams),))
    
    beta0 = coefs[:nteams]
    beta_player = coefs[nteams:(nplayers + nteams)]
    gamma0 = coefs[(nplayers + nteams):(nplayers + 2*nteams)]
    gamma_player = coefs[(nplayers + 2*nteams):]
    
    # parameterize sigma by its log
    #logsigma = coefs[-1]
    
    assert(beta0.shape == (nteams,))
    assert(beta_player.shape == (nplayers,))
    assert(beta0.shape == (nteams,))
    assert(gamma_player.shape == (nplayers,))
    
    
    return (beta0, beta_player, gamma0, gamma_player)#, logsigma)

# Use a different log-likelihood
@pm.observed(name = "loglik", observed = True)
def loglik(coefs = coefs, tau = tau, value = score_diff, 
            off_rating = off_rating, def_rating = def_rating, 
            nplayers = player_num, nteams = team_num,
            guest_matrix = guest_matrix, host_matrix = host_matrix, guest_lineup_arr = guest_lineup_arr,
            host_lineup_arr = host_lineup_arr):

    beta0, betas, gamma0, gammas =              split_params(coefs, nplayers, nteams)
    
    ngames = value.shape[0]
    
    guest_off_0 = np.dot(guest_matrix, beta0)
    guest_def_0 = np.dot(guest_matrix, gamma0)
    host_off_0 = np.dot(host_matrix, beta0)
    host_def_0 = np.dot(host_matrix, gamma0)
    
    guest_off = guest_off_0 + np.dot(guest_lineup_arr, betas * off_rating)
    guest_def = guest_def_0 + np.dot(guest_lineup_arr, gammas * def_rating)
    host_off = host_off_0 + np.dot(host_lineup_arr, betas * off_rating)
    host_def = host_def_0 + np.dot(host_lineup_arr, gammas * def_rating)
    
    mean = guest_off - host_def - (host_off - guest_def)
    
    loglik = pm.normal_like(value, mean, tau)
#     loglik -= lam * np.linalg.norm(coefs[:-1])
    return(loglik)

parameterlist = [loglik, coefs, tau]
response_model2=pm.Model(parameterlist)

mcmc2 = pm.MCMC(response_model2)

mcmc2.sample(iter=1000000, burn = 0, thin = 1)

coefs.trace()

geweke_res = pm.geweke(coefs.trace()[99900:,50])
pm.Matplot.geweke_plot(geweke_res,'coefs')

