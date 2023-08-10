import sys
sys.maxsize

sys.maxsize.bit_length()

2**63 - sys.maxsize

t = sys.maxsize**10

t.bit_length()

M49 = 2**74207281 -1

M49.bit_length()

## Option is the right to buy or sell a stock

from math import log, sqrt, exp

from scipy import stats

def bs_call_value(S,K,T,r,sigma):
    d1 = (log(S/K) + (r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    
    N_d1 = stats.norm.cdf(d1,0.0,1.0)
    N_d2 = stats.norm.cdf(d2,0.0,1.0)
    
    call_price = (S*N_d1 - K*exp(-r * T) * N_d2)
    
    return call_price

S = 89.0
K = 100.0
T = 0.5
r = 0.02
sigma = 0.2

call_price = bs_call_value(S,K,T,r,sigma)
call_price

print("This European call price is: ${:20.16f}".format(call_price))

##20 digits and 16 digits after dot

S = 102.5
K = 88.5
r = 0.03
T = 0.25
sigma = 0.3

def bs_call_put(S,K,T,r,sigma):
    d1 = (log(S/K) + (r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    
    N_d1 = stats.norm.cdf(-d1,0.0,1.0)
    N_d2 = stats.norm.cdf(-d2,0.0,1.0)
    
    put_price = -S*N_d1 + K*exp(-r * T) * N_d2
    
    return put_price

put_price = bs_call_put(S,K,T,r,sigma)

print("This European put price is: $%f"%(put_price))

#dS = rSdt + &Sdz
## dz is normal distribution
## The little change in a stock price at time T is a random walk times volatility and current price

from numpy import *

random.seed(1000)
N = 100000
z = random.standard_normal(N)

for i in range(0,20):
    print("{:15.8f}".format(z[i]))

def mc_simulation_eu_call(s, K, T, r, no_trial):
    # Input:
    ##s: initial input
    ##K: strike price
    ## T: time to maturity time
    ## r: riskless interest rate
    ## no_trial: simulation steps
    
    ##Output:
    ## european call price
    
    random.seed(10000)
    z = random.standard_normal(no_trial)
    ## stock price at Time T
    ST = s*exp(r*T+sigma*sqrt(T)*z)
    payoff = maximum(ST - K,0)
    eu_call_price = exp(-r*T)*sum(payoff)/no_trial
    
    return eu_call_price

s = 89.0
K = 102.0
T = 0.5
r = 0.03
sigma = 0.3
no_trial = 1000000

mc_simulation_eu_call(s, K, T, r, no_trial)

def mc_simulation_eu_put(s, K, T, r, no_trial):
    # Input:
    ##s: initial input
    ##K: strike prive
    ## T: time to maturity time
    ## r: riskless interest rate
    ## no_trial: simulation steps
    
    ##Output:
    ## european call price
    
    random.seed(10000)
    z = random.standard_normal(no_trial)
    ## stock price at Time T
    ST = s*exp(r*T+sigma*sqrt(T)*z)
    payoff = maximum(K - ST,0)
    eu_put_price = exp(-r*T)*sum(payoff)/no_trial
    
    return eu_put_price

