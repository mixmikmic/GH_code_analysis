import numpy as np

def simu_coin_toss():
    """Function to simulate a coin toss
    Returns:
        number of tails until first head
    """
    num_tails = 0
    while True:
        head = np.random.uniform(size=1) > 0.5
        #res.append(head[0])
        if not head: # tails
            num_tails += 1
        else:  # heads
            break
    return num_tails

def make_Xk(num_tails):
    """ Make a sequence of num_tails+1 uniform random variables"""
    X_k = np.random.uniform(low=0, high=3, size=num_tails+1)
    return X_k

def make_X(num_tails):
    """ Make r.v. X which is the sum of num_tails+1 uniform r.vs"""
    X_k = make_Xk(num_tails)
    X = np.sum(X_k)
    return X

X_array = []
for i_sim in range(10000):
    num_tails = simu_coin_toss()
    
    X_k = make_Xk(num_tails)
    X = np.sum(X_k)
    #print({"{0}: {1}: {2}".format(num_tails, X_k, X)})
    X_array.append(X)
    
print("{Mean of r$X$: {0}}".format(np.mean(X_array))) 
print("{Mean of r$X$: {0}}".format(np.mean(X_array))) 

