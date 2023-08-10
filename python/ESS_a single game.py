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

import re
import numpy as np
import cPickle as pickle
# import pickle as pickle

player_info = pd.read_csv("clean-data/player_info_pergame.csv")
game_outcomes = pd.read_csv("clean-data/game_outcomes_15-16.csv")

N_players = len(player_info)
N_teams = np.max(game_outcomes['Visitor_Index'].values) + 1
N_games =  len(game_outcomes)

print N_players, N_teams, N_games
print 2 * (N_players + N_teams)

with open("clean-data/host_team_line_up.pkl", "rb") as f:
    host_team_line_up = pickle.load(f)

with open("clean-data/guest_team_line_up.pkl", "rb") as f:
    guest_team_line_up = pickle.load(f)

################# INPUT SECTION ###################
###################################################

########### player indicator for each game ###############
### host_lineup_arr & guest_lineup_arr are (1230 x 476)###
with open("clean-data/host_team_line_up.pkl", "rb") as f:
    host_team_line_up = pickle.load(f)

with open("clean-data/guest_team_line_up.pkl", "rb") as f:
    guest_team_line_up = pickle.load(f)
    
# print len(host_team_line_up[0])
host_lineup_arr = np.array(host_team_line_up)
guest_lineup_arr = np.array(guest_team_line_up)


######### team indicator for each game ########### 
### host_matrix & guest_matrix are (1230 x 30) ###
guest_matrix = np.zeros((game_outcomes.shape[0], np.max(game_outcomes['Visitor_Index']) + 1), dtype = bool)
guest_matrix.shape
host_matrix = np.copy(guest_matrix)

def make_matrix(mat, indices):
    for (i, ind) in enumerate(indices):
        mat[i, ind] = True

make_matrix(host_matrix, game_outcomes['Visitor_Index'].values)
make_matrix(guest_matrix, game_outcomes['Home_Index'].values)


############## Observed data ##################
score_diff = game_outcomes['diff'].values
off_rating = player_info['PTS'].values + player_info['AST'].values
def_rating = player_info['BLK'].values + player_info["STL"].values + player_info['DRB'].values

def diff_score_calc(f_guest, f_host, game_index):
    # f_i is the state of the latent variables: the vector 'beta' for team2 and the vector 'gamma' for team2
    guest_i = guest_team_line_up[game_index]
    host_i = host_team_line_up[game_index]
    
    guest_diff = (f_guest[0] + off_rating[guest_i].dot(f_guest[1:])) - (f_host[0] + def_rating[host_i].dot(f_host[1:]))
    host_diff = (f_host[0] + off_rating[host_i].dot(f_host[1:])) - (f_guest[0] + def_rating[guest_i].dot(f_guest[1:]))
    return guest_diff - host_diff    

# log likelihood for standard Gaussian
# The standard deviation for the Gaussian was determined to be 10-ish basedon the distribution of the actual data
def log_likelihood(f_guest, f_host, game_index):
    new_mean = diff_score_calc(f_guest, f_host, game_index)    
    return scipy.stats.norm.logpdf(score_diff[game_index], new_mean, 10.0)

# Model covariance matrix
# We assigned covaraince randomly sampled from the gamma distribution (to make sure it's non-negative)
def covariance(dim, gammaMean = 1.0, gammaScale = 1.0):
    mat = np.identity(dim)
    for i in np.arange(dim):
        for j in np.arange(dim):
            if (i == j):
                continue
            mat[i,j] = np.random.gamma(gammaMean, gammaScale, size=None)
    return mat
# print correlation()[2*N_teams + N_players][2 * N_teams + N_players:2 * N_teams + N_players +N_players]

mvn = np.random.multivariate_normal
def ess(game_index, log_likelihood, N_mcmc, burn_in):
    ## 1 indicates 'guest'
    ## 2 indicates 'host'
    N_1 = len(player_info[guest_team_line_up[game_index]]) + 1
    N_2 = len(player_info[host_team_line_up[game_index]]) + 1
    
    # INITIALIZATION
    # Initial proposals are drawn from the standard Gausian
    mcmc_samples_1 = np.random.randn(N_mcmc + burn_in, N_1)
    mcmc_samples_2 = np.random.randn(N_mcmc + burn_in, N_2)
    
    # random draw from normal distribution with which we'll determine
    # new state of the latent variables
    norm_samples_1 = mvn(np.zeros(N_1), covariance(N_1), N_mcmc + burn_in)
    norm_samples_2 = mvn(np.zeros(N_2), covariance(N_2), N_mcmc + burn_in)
    
    # random draw from unifrom distribution with which we'll determine 
    # the loglikelihood threshold (likelihood threshold defines the 'slice' where we sample)
    unif_samples = np.random.uniform(0, 1, N_mcmc+burn_in)
    
    # initial proposal of the theta
    theta = np.random.uniform(0, 2*np.pi, N_mcmc+burn_in)
    
    # variables with which we'll propose a new state by shrinking the range of theta
    theta_min = theta - 2*np.pi
    theta_max = theta + 2*np.pi
    
    # We select a new location (i.e. new state of the latent variables)
    # on the randomly generated ellipse given theta and norm_samples
    for i in range(1, N_mcmc + burn_in):
#         if i % 100 == 0:
#             print i

        # initial state of the latent vairables    
        f_1 = mcmc_samples_1[i - 1,:]
        f_2 = mcmc_samples_2[i - 1,:]
        #print f, data
        
        # the loglikelihood threshold
        # the threshold is chosen between [0, Likelihood]
        llh_thresh = log_likelihood(f_1, f_2, game_index) + np.log(unif_samples[i])
        
        f_prime_1 = (f_1 * np.cos(theta[i])) + (norm_samples_1[i,:] * np.sin(theta[i]))
        f_prime_2 = (f_2 * np.cos(theta[i])) + (norm_samples_2[i,:] * np.sin(theta[i]))
        while log_likelihood(f_prime_1, f_prime_2, game_index) < llh_thresh:
            if theta[i] < 0:
                theta_min[i] = theta[i]
            else:
                theta_max[i] = theta[i]
                
            theta[i] = np.random.uniform(theta_min[i], theta_max[i], 1)  
            f_prime_1 = (f_1 * np.cos(theta[i])) + (norm_samples_1[i,:]*np.sin(theta[i]))
            f_prime_2 = (f_2 * np.cos(theta[i])) + (norm_samples_2[i,:]*np.sin(theta[i]))
            
        mcmc_samples_1[i,:] = f_prime_1
        mcmc_samples_2[i,:] = f_prime_2
    
    return mcmc_samples_1[(burn_in+1):(burn_in+N_mcmc),], mcmc_samples_2[(burn_in+1):(burn_in+N_mcmc),]

game_index = 10
N_mcmc = 40000
burn_in = 500
start_time = time.time()
guest_states, host_states = ess(game_index, log_likelihood, N_mcmc, burn_in)
# print guest_states
# print guest_states.shape
elapsed = time.time() - start_time
print("Elapsed Time (sec): %f" %elapsed)

mean_g = guest_states.mean(axis=0)
mean_h =  host_states.mean(axis=0)
print "The mean beta coefficients for each player in the Guest team of the 10th game:"
print mean_g, mean_g.shape
print "--------------------------------------------------------"
print "The mean beta coefficients for each player in the Host team of the 10th game:"
print mean_h, mean_h.shape
print ""
print ""
print ""
print ""

real = score_diff[game_index]
estimated = diff_score_calc(mean_g, mean_h, game_index)
print 'the actual differential score = %s' %(real)
print 'the differential score with the estimated variable states = %s' %(estimated)
print 'difference is %s percent' %(np.fabs(real - estimated)*100.0/real)

for i in range(10,11):
    plt.figure(i)
    p_hist = plt.hist(guest_states[burn_in:,i], bins=100)
    plt.title("Guest variable states in %sth game" %(game_index))
    plt.xlabel("$beta^{O}$ for the %sth player" %(i+1))
    plt.ylabel("counts")
    p_map_index = np.argmax(p_hist[0])
    p_hist_bin_middle = 0.5*p_hist[1][:-1] + 0.5*p_hist[1][1:]
    p_map = p_hist_bin_middle[p_map_index]
    
    print "The MAP value for beta %s is: %s" %(i, p_map)

for i in range(10,11):
    plt.figure(i)
    p_hist = plt.hist(host_states[burn_in:,i], bins=100)
    plt.title("Host variable states in the %s th game" %(game_index))
    plt.xlabel("$beta^{D}$ of the %sth player" %(i+1))
    plt.ylabel("counts")
    p_map_index = np.argmax(p_hist[0])
    p_hist_bin_middle = 0.5*p_hist[1][:-1] + 0.5*p_hist[1][1:]
    p_map = p_hist_bin_middle[p_map_index]
    
    print "The MAP value for beta %s is: %s" %(i, p_map)

N_mcmc = 1500
burn_in = 80

start_time = time.time()
temp = np.zeros(N_games)
for i in np.arange(N_games):
    guest_states, host_states = ess(i, log_likelihood, N_mcmc, burn_in)
    mean_g = guest_states.mean(axis=0)
    mean_h =  host_states.mean(axis=0)

    real = score_diff[i]
    estimated = diff_score_calc(mean_g, mean_h, i)
    temp[i] = np.fabs(real - estimated)*100.0/real

elapsed = time.time() - start_time
print("Elapsed Time (sec): %f" %elapsed)

print "The averaged difference between the observed and esimated differential scores: %s percent" %(np.fabs(np.mean(temp)))
print "standard deviation: %s percent" %(np.std(temp))

p_hist = plt.hist(temp, bins=150)
plt.title("Difference b/w the estimated and actual differential scores")
plt.xlabel("%")
plt.ylabel("counts")



