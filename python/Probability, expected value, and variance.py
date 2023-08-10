# We'll need numpy's random number generator
import numpy as np

# Initialize counters for number of people of each type
sick_and_pos = 0
sick_and_neg = 0
healthy_and_pos = 0
healthy_and_neg = 0

# Simulate 2000 patients
for i in range(2000):
    sick = (np.random.rand() < 0.01)
    if sick:
        test = (np.random.rand() < 0.99)
        if test:
            sick_and_pos += 1
        else:
            sick_and_neg += 1
    else:
        test = (np.random.rand() < 0.05)
        if test:
            healthy_and_pos += 1
        else:
            healthy_and_neg += 1

print 'Empirical P(A|B):', float(sick_and_pos)/(sick_and_pos + healthy_and_pos)

# Initialize counters for # games played and amount earned
fair_games = 0
fair_money = 0.0
biased_games = 0
biased_money = 0.0

# Play 300 fair games
for i in range(300):
    flip = (np.random.rand() < 0.5) # Simulating 50% probability of heads
    fair_games += 1
    if flip:
        fair_money += 2
    else:
        fair_money -= 1
    
print 'Fair games played:', fair_games, 'Money:', fair_money, '$/games:', fair_money/fair_games

# Play 300 biased games (with the biased coin)
for i in range(300):
    flip = (np.random.rand() < 0.333333) # Simulating 1/3 probability of heads
    biased_games += 1
    if flip:
        biased_money += 2
    else:
        biased_money -= 1
    
print 'Biased games played:', biased_games, 'Money:', biased_money, '$/games:', biased_money/biased_games

