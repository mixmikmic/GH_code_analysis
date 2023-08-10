import numpy as np

# Initialize the outcomes
success = 0                 # 2 gold coins found
good_trials = 0             # 1 gold coin found (1st condition satisfied)

# Initialize the boxes as
box1 = [0,0];               # 2 silver coins   
box2 = [0,1];               # 1 silver and 1 gold coin
box3 = [1,1];               # 2 gold coins

# The number of times we want to repeat the event
n_trials = 10000

for i in range(0, n_trials):
    # This will explicitly hold the value of the coin that 
    # was picked during each trial.
    pick1 = 0;
    pick2 = 0;
    
    # Randomly select a number [0, 2]. This is the box we have selected.
    box_pick = np.random.randint(1, high=4, size=1);

    # If box 1 was picked then both coins are silver.
    if(box_pick==1):
        pick1 = 0;
        pick2 = 0;
        
    # If box 2 was picked
    if(box_pick==2):
        # Randomly select a number in [0, 1] = [Silver, Gold] 
        pick1 = box2[np.random.randint(0, high=2, size=1)[0]];
        # If a gold coin was picked then the second coin will be silver
        # The first condition has been met.
        if(pick1==1):
            pick2 = 0;
            good_trials = good_trials + 1;
        # If a silver coin was picked then the second coin will be gold
        # The first condition has not been met.
        else:
            pick2 = 1;
     
    # If box 3 was picked then both coins are gold
    # The first condition will alsways be met.
    if(box_pick==3):
        pick1 = 1;
        pick2 = 1;
        good_trials = good_trials + 1;
    
    # If both coins picked were gold then this is a successful trial.
    # It satisfies the second condition.
    if(pick1 ==1 and pick2 == 1):
        success = success + 1
        
# Print the probability of both coins being gold given the first ball
# picked was gold.
print(success/good_trials)        

import random
import numpy as np
import matplotlib.pyplot as plt

num_tries = [1,2,5,10,15,25,50,75,100,250,500,1000,10000, 100000, 1000000];
results = np.zeros((len(num_tries), 1));

for j in range(0, len(num_tries)):
    # Initialize the outcomes
    success = 0                 # 2 gold coins found
    good_trials = 0             # 1 gold coin found (1st condition satisfied)

    # Initialize the boxes as
    box1 = [0,0];               # 2 silver coins   
    box2 = [0,1];               # 1 silver and 1 gold coin
    box3 = [1,1];               # 2 gold coins

    for i in range(0, num_tries[j]):
        # This will explicitly hold the value of the coin that 
        # was picked during each trial.
        pick1 = 0;
        pick2 = 0;
    
        # Randomly select a number [0, 2]. This is the box we have selected.
        box_pick = np.random.randint(1, high=4, size=1);

        # If box 1 was picked then both coins are silver.
        if(box_pick==1):
            pick1 = 0;
            pick2 = 0;
        
        # If box 2 was picked
        if(box_pick==2):
            # Randomly select a number in [0, 1] = [Silver, Gold] 
            pick1 = box2[np.random.randint(0, high=2, size=1)[0]];
            # If a gold coin was picked then the second coin will be silver
            # The first condition has been met.
            if(pick1==1):
                pick2 = 0;
                good_trials = good_trials + 1;
                # If a silver coin was picked then the second coin will be gold
                # The first condition has not been met.
            else:
                pick2 = 1;
     
        # If box 3 was picked then both coins are gold
        # The first condition will alsways be met.
        if(box_pick==3):
            pick1 = 1;
            pick2 = 1;
            good_trials = good_trials + 1;
    
        # If both coins picked were gold then this is a successful trial.
        # It satisfies the second condition.
        if(pick1 == 1 and pick2 == 1):
            success = success + 1
    
    if not good_trials == 0:
        results[j] = success/good_trials
    else:
        results[j] = 0

plt.plot(np.log(num_tries), results)
plt.ylim(0, 1.0)
plt.show()

import random
import numpy as np
import matplotlib.pyplot as plt

num_tries = [1,2,5,10,15,25,50,100,500,1000,10000, 100000, 1000000];
#num_tries = [1000];

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x = [1,2,3,4]

results = np.zeros((len(num_tries), 1));

for j in range(0, len(num_tries)):
    success = 0
    trials = 0
    for i in range(0,num_tries[j]):
        trials += 1
        y = np.random.permutation(len(x)) + 1
        
        if not 0 in x-y:
            success += 1
    
    results[j] = success/trials

print(results[-1])

plt.plot(np.log(num_tries), results)
plt.ylim(0, 1.0)
plt.show()

import random
import numpy as np
import matplotlib.pyplot as plt

num_tries = [1,2,5,10,15,25,50,100,500,1000,10000, 100000, 1000000];
#num_tries = [1000];

results = np.zeros((len(num_tries), 1));

for j in range(0, len(num_tries)):
    
    decision = 'switch' # 'switch'
    num_doors = 3
    
    success = 0
    trials = 0
    
    for i in range(0,num_tries[j]):
        trials += 1
        
        doors = np.zeros((num_doors,1))
        doors[np.random.randint(0, high=num_doors, size=1)[0]] = 1
        
        door_picked = np.random.randint(0, high=num_doors, size=1)
        
        if(decision == 'stay'):
            if(doors[door_picked] == 1):
                success += 1
                
        if(decision == 'switch'):
            if(not doors[door_picked] == 1):
                success += 1        
    
    results[j] = success/trials

print(results[-1])

plt.plot(np.log(num_tries), results)
plt.ylim(0, 1.0)
plt.show()



