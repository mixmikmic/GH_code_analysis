import random
from numba import jit

# Monte Carlo simulation function. This is defined as
# a function so the numba library can be used to speed
# up execution. Otherwise, this would run much slower.
# p1 is the probability of the first area, and s1 is the
# score of the first area, and so on. The probabilities
# are cumulative.
@jit
def MCHist(n_hist, p1, s1, p2, s2, p3, s3, p4, s4):
    money = 0
    for n in range(1, n_hist):
        x = random.random()       
        if x <= p1:
            money += s1
        elif x <= (p1 + p2):
            money += s2
        elif x <= (p1 + p2 + p3):
            money += s3
        elif x <= (p1 + p2 + p3 + p4):
            money += s4
    return money

# Run the simulation, iterating over each number of 
# histories in the num_hists array. Don't cheat and look 
# at these probabilities!! "You" don't know them yet.
num_hist = 1e3 # $500
results = MCHist(num_hist, 0.05, 1, 0.3, 0.3, 0.15, 0.5, 0.5, 0.2) 
payout = round(results / num_hist, 3)
print('Expected payout per spin is ${}'.format(payout))

num_hist2 = 1e8 # $50 million
results2 = MCHist(num_hist2, 0.05, 1, 0.3, 0.3, 0.15, 0.5, 0.5, 0.2) 
payout2 = round(results2 / num_hist2, 3)
print('Expected payout per spin is ${}'.format(payout2))

num_hist3 = 1e3 # $500
results3 = MCHist(num_hist3, 0.25, 0.2, 0.25, 0.36, 0.25, 0.3, 0.25, 0.4) 
payout3 = round(results3 / num_hist3, 5)
print('Expected payout per spin is ${}'.format(payout3))

num_hist4 = 1e3 # $500
results4 = MCHist(num_hist4, 0.159, 0.315, 0.286, 0.315, 0.238, 0.315, 0.317, 0.315) 
payout4 = round(results4 / num_hist4, 3)
print('Expected payout per spin is ${}'.format(payout4))

