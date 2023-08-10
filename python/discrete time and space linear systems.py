import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)
get_ipython().run_line_magic('matplotlib', 'inline')


def get_eigs(T):
    T = np.array(T)
    eigvals, eigvecs = np.linalg.eig(T)

    # We want to normalize the eigenvectors - make them sum to 1.
    # So, sum down the columns, then divide each column by its sum
    col_sums = eigvecs.sum(axis=0, keepdims=True)

    # Problem - some of those columns sum could be 0.  We can't divide by 0.  So, we protect ourselves by:
    # Finds columns sums that are close to 0
    tol = 0.001
    idx = np.abs(col_sums) < tol
    # Now put 1 in each of those spots.  When you divide by 1, nothing happens.
    #So, a column that sums to 0 will have nothing done to it.
    col_sums[idx] = 1
    eigvecs = eigvecs / col_sums
    return eigvals, eigvecs

# Mars colonies

T = [[0.65, 0.00, 0.30],
     [0.20, 0.90, 0.00],
     [0.15, 0.10, 0.70]
    ]

T = np.array(T)
n = len(T)   # Detect number of states

eigvals, eigvecs = get_eigs(T)

print(eigvals)
i = 0
val = eigvals[i]
good_vec = eigvecs[:,i]
print("Eigen analysis says stationary distribution should be")
print(good_vec)

print("\n Let run it and check")
# Inital distribution is arbitrary.  Here are two commonly used ones.  

# This starts everyone at state k
k = 3
k = min(k, n-1)  # States are 0, 1, 2, ..., n-1.  If you pick k > n-1, this fixes your error.
x = np.zeros(n)
x[k] = 1.0

# This makes it uniform across all states.
# x = np.ones(n)/n

print(x)
x_result = [x]
Steps = 100

for i in range(Steps):
    x = T.dot(x)
    x_result.append(x)
    
x_result = np.array(x_result)
print(x_result)
print("Eigen analysis says it should be")
print(good_vec)
print("Do they match?")

# Growing shrub

r = 7
T = [[1, 1],
     [r, 0],
    ]

T = np.array(T)
n = len(T)   # Detect number of states

eigvals, eigvecs = get_eigs(T)

print(eigvals)
i = 0
val = eigvals[i]
good_vec = eigvecs[:,i]
print("Eigen analysis says stationary distribution should be")
print(good_vec)

print("\n Let run it and check")
print("Start with 0 mature and 1 young branch.")
x = np.array([0,1.0])
print(x)
x_result = [x]
Steps = 100

for i in range(Steps):
    x = T.dot(x)
    x_result.append(x)
x_result = np.array(x_result)
# print(x_result)
row_sums = x_result.sum(axis=1, keepdims=True)
x_result_normalized = x_result / row_sums
print(x_result_normalized)
print("Eigen analysis says it should be")
print(good_vec)
print("Do they match?")

# Naive Google PageRank

# I find it easier to enter the matrix correctly if put all OUT arrows for state i into ROW i.
# But that should be COLUMN i.  So, we do the tranpose.

adj = [[0, 0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 0, 1]
      ]
A = np.array(adj)
A = A.T
col_sums = A.sum(axis=0, keepdims=True)
T = A / col_sums
print(T)
n = len(T)   # Detect number of states


eigvals, eigvecs = get_eigs(T)

print(eigvals)
i = 0
val = eigvals[i]
good_vec = eigvecs[:,i]
print("Eigen analysis says stationary distribution should be")
print(good_vec)

print("\n Let run it and check")
# This makes it even across all states.
x = np.ones(n)/n

print(x)
x_result = [x]
Steps = 100

for i in range(Steps):
    x = T.dot(x)
    x_result.append(x)
x_result = np.array(x_result)
row_sums = x_result.sum(axis=1, keepdims=True)
x_result_normalized = x_result / row_sums
print(x_result_normalized)
print("Eigen analysis says it should be")
print(good_vec)
print("Do they match?")

# Wise Google PageRank

# I find it easier to enter the matrix correctly if put all OUT arrows for state i into ROW i.
# But that should be COLUMN i.  So, we do the tranpose.

adj = [[0, 0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 0, 1]
      ]
A = np.array(adj)
A = A.T
col_sums = A.sum(axis=0, keepdims=True)
T = A / col_sums
n = len(T)   # Detect number of states
p = 0.9
q = 1 - p
T = T*p + q/n

print(T)

eigvals, eigvecs = get_eigs(T)

print(eigvals)
i = 1
val = eigvals[i]
good_vec = eigvecs[:,i]
print("Eigen analysis says stationary distribution should be")
print(good_vec)

print("\n Let run it and check")
# This makes it even across all states.
x = np.ones(n)/n

print(x)
x_result = [x]
Steps = 100

for i in range(Steps):
    x = T.dot(x)
    x_result.append(x)
x_result = np.array(x_result)
row_sums = x_result.sum(axis=1, keepdims=True)
x_result_normalized = x_result / row_sums
print(x_result_normalized)
print("Eigen analysis says it should be")
print(good_vec)
print("Do they match?")

# Naive Google PageRank


# I find it easier to enter the matrix correctly if put all OUT arrows for state i into ROW i.
# But that should be COLUMN i.  So, we do the tranpose.

adj = [[0, 0, 1, 1, 1, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 0, 1]
      ]
A = np.array(adj)
A = A.T
col_sums = A.sum(axis=0, keepdims=True)
T = A / col_sums
print(T)
n = len(T)   # Detect number of states

eigvals, eigvecs = get_eigs(T)
print(eigvals)

i = [0,1,2]
val = eigvals[i]
vec = eigvecs[:,i]  #recall, eigenvectors go DOWN columns, not across rows
print(vec)
print()
print(T.dot(vec))
print()
print(T.dot(T.dot(vec)))

print("\n Let run it and check")

# Inital distribution MATTERS

def run_trial(x):
    x = np.array(x)
    x_result = [x]
    Steps = 100

    for i in range(Steps):
        x = T.dot(x)
        x_result.append(x)
    x_result = np.array(x_result)
    row_sums = x_result.sum(axis=1, keepdims=True)
    x_result_normalized = x_result / row_sums
    print(x_result_normalized)

print("Run with initial distribution = column 0")
x = [0, 0, 0.5, 0.5, 0, 0]
run_trial(x)
print("Eigen analysis says it should be")
print(eigvecs[:,0])
print("Do they match?")


print()
print("Run with initial distribution = column 2")
x = [0, 0, 0, 0, 0, 1.0]
run_trial(x)
print("Eigen analysis says it should be")
print(eigvecs[:,2])
print("Do they match?")


print()
print("Run with initial distribution = everyone at 2")
x = [0, 0, 1.0, 0, 0, 0.0]
run_trial(x)
print("Eigen analysis says it should oscilate between states 2 & 3")
print("Do they match?")


# Kai-Quinten Bunk Bed Hopping

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)
get_ipython().run_line_magic('matplotlib', 'inline')

# state = [where Kai sleeps tonight, where Quinten sleeps tonight]

# state 0 = [bottom, bottom]
# state 1 = [top, bottom]
# state 2 = [top, top]
# state 3 = [bottom, top]

# I find it easier to enter the matrix correctly if put all OUT arrows for state i into ROW i.
# But that should be COLUMN i.  So, we do the tranpose.

adj = [[0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       [1, 0, 0, 0],
      ]
A = np.array(adj)
A = A.T
col_sums = A.sum(axis=0, keepdims=True)
T = A / col_sums
n = len(T)   # Detect number of states

print(T)

eigvals, eigvecs = get_eigs(T)

print(eigvals)
i = 3
val = eigvals[i]
good_vec = eigvecs[:,i]
print("Eigen analysis says stationary distribution should be")
print(good_vec)

print("\n Let run it and check")
print("Let's start with [bottom, bottom]")
print(x)
x_result = [x]
Steps = 100

for i in range(Steps):
    x = T.dot(x)
    x_result.append(x)
x_result = np.array(x_result)
row_sums = x_result.sum(axis=1, keepdims=True)
x_result_normalized = x_result / row_sums
print(x_result_normalized)
print("Eigen analysis says it should be")
print(good_vec)
print("Do they match?")

print("\n Let run it and check")

# Inital distribution MATTERS

def run_trial(x):
    x = np.array(x)
    x_result = [x]
    Steps = 100

    for i in range(Steps):
        x = T.dot(x)
        x_result.append(x)
    x_result = np.array(x_result)
    row_sums = x_result.sum(axis=1, keepdims=True)
    x_result_normalized = x_result / row_sums
    print(x_result_normalized)

print("Run with uniform distribution")
x = [1/4, 1/4, 1/4, 1/4]
run_trial(x)
print("Eigen analysis says it should be")
print(eigvecs[:,3])
print("Do they match?")


print()
print("Run with initial distribution all on state 0")
x = [1, 0, 0, 0]
run_trial(x)
print("Eigen analysis says it should cycle among the 4 states.  It has period = 4")
print("Do they match?")


print()
print("Run with initial distribution equally on states 0 and 2")
x = [1/2, 0, 1/2, 0]
run_trial(x)
print("Eigen analysis says it should cycle among (0,2) and (1,3).  It has period = 2")
print("Do they match?")


print()
print("Run with initial distribution equally on 0, 1, 2")
x = [1/3, 1/3, 1/3, 0]
run_trial(x)
print("Eigen analysis says it should cycle where 1 state is empty, rest are equal.  It has period = 4")
print("Do they match?")

print("Analyze 1st eigval")
v = eigvals[0]
print(v)
print("Nope, that's -1.  We want +1")
print(v**2)
print("Done.  There is a stable oscillation with period 2")

print()
print("Analyze 2nd eigval")
v = eigvals[1]
print(v)
print("Nope, that's +i.  We want +1")
print(v**2)
print("Nope, that's -1.  We want +1)")
print(v**3)
print("Nope, that's -i.  We want +1")
print(v**4)
print("Done.  There is a stable oscillation with period 4")

print()
print("Analyze 3rd eigval")
v = eigvals[2]
print(v)
print("Nope, that's -i.  We want +1")
print(v**2)
print("Nope, that's -1.  We want +1)")
print(v**3)
print("Nope, that's +i.  We want +1")
print(v**4)
print("Done.  There is another stable oscillation with period 4")

print()
print("Analyze 4th eigval")
v = eigvals[3]
print(v)
print("Done.  There is a stable oscillation with period 1 ... equilibrium.")

