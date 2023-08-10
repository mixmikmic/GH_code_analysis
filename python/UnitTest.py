#Persistence Filter files
from libpython_persistence_filter import *
from libpython_persistence_filter_utils import *

from numpy import *

#Set up survival-time priors
lambda_u = 1
lambda_l = .01
logS_T = lambda t : log_general_purpose_survival_function(t, lambda_l, lambda_u)
S_T = lambda t : exp(logS_T(t))

#Set up observation models
P_M = .2
P_F = .01

#Set up Persistence Filter
pf = PersistenceFilter(logS_T, 0.0)

#FIRST OBSERVATION:  y_1 = 0 at time t_1 = 1.0

#Update the filter
t_1 = 1.0
pf.update(False, t_1, P_M, P_F)
#Posterior prediction
filter_posterior1 = pf.predict(t_1)

#Compute ground-truth value by hand:

#The likelihood p(y_1 = 0 | T >= t_1) = P_M.
pY1_t1 = P_M
  
#The evidence probability p(y_1 = 0) = p(y_1 = 0 | T >= t_1) * p(T >= t_1) + p(y_1 = 0 | T < t_1) * p(T < t_1)
pY1 = P_M * S_T(t_1) + (1 - P_F) * (1 - S_T(t_1))

#Compute the posterior p(X_{t_1} = 1 | y_1 = 0) = p(y_1 = 0 | T >= t_1) / p(y_1 = 0) * p(T >= t_1)
posterior1 = (pY1_t1 / pY1) * S_T(t_1)

print "FILTER STATE AFTER INCORPORATING y_1 = 0 at time t_1 = 1.0"
print "Filter posterior probability p(X_{t_1} = 1 | y_1 = 0) = %g" % filter_posterior1
print "True posterior probability p(X_{t_1} = 1 | y_1 = 0) = %g\n" % posterior1


#SECOND OBSERVATION:  y_2 = 1 at time t_1 = 2.0

#Update the filter
t_2 = 2.0
pf.update(True, t_2, P_M, P_F)
#Posterior prediction
filter_posterior2 = pf.predict(t_2)

#Compute ground-truth value by hand

#The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
pY2_t2 = P_M * (1-P_M)

# The evidence probability p(y_1 = 0, y_2 = 1) = 
# p(y_1 = 0, y_2 = 1 | T > t_2) * p(T > t_2) +
# p(y_1 = 0, y_2 = 1 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
# p(y_1 = 0, y_2 = 1 | T < t_1) * p(t < t_1)

pY2 = P_M * (1 - P_M) * S_T(t_2) + P_M * P_F * (S_T(t_1) - S_T(t_2)) + (1 - P_F) * P_F * (1 - S_T(t_1))

#Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1) = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)
posterior2 = (pY2_t2 / pY2) * S_T(t_2)


print "FILTER STATE AFTER INCORPORATING y_2 = 1 at time t_2 = 2.0"
print "Filter posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = %g" % filter_posterior2
print "True posterior probability p(X_{t_2} = 1 | y_1 = 0, y_2 = 1) = %g\n" % posterior2



#THIRD OBSERVATION:  y_3 = 0 at time t_3 = 3.0

#Update the filter
t_3 = 3.0
pf.update(False, t_3, P_M, P_F)
#Posterior prediction
filter_posterior3 = pf.predict(t_3)

#Compute ground-truth-value by hand

#The likelihood p(y_1 = 0, y_2 = 1 | T >= t_2)
pY3_t3 = P_M * (1-P_M) * P_M;
  
#The evidence probability p(y_1 = 0, y_2 = 1, y_3 = 0) = 
# p(y_1 = 0, y_2 = 1, y_3 = 0 | T > t_3) * p(T > t_3) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | t_2 <= T < t_3) * p(t_2 <= T < t_3) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | t_1 <= T < t_2) * p(t_1 <= T < t_2) +
# p(y_1 = 0, y_2 = 1, y_3 = 0 | T < t_1) * p(t < t_1)
  
pY3 = P_M * (1 - P_M) * P_M * S_T(t_3) + P_M * (1 - P_M) * (1 - P_F) * (S_T(t_2) - S_T(t_3)) + P_M * P_F * (1 - P_F) * (S_T(t_1) - S_T(t_2)) + (1 - P_F) * P_F * (1 - P_F) * (1 - S_T(t_1))
    
#Compute the posterior p(X_{t_2} = 2 | y_1 = 0, y_2 = 1) = p(y_1 = 0, y_2 = 1 | T >= t_2) / p(y_1 = 0, y_2 = 1) * p(T >= t_2)
posterior3 = (pY3_t3 / pY3) * S_T(t_3)


print "FILTER STATE AFTER INCORPORATING y_3 = 0 at time t_3 = 3.0"
print "Filter posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = %g" % filter_posterior3
print "True posterior probability p(X_{t_3} = 1 | y_1 = 0, y_2 = 1, y_3 = 0) = %g\n" % posterior3



