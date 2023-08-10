from scipy.stats import chisquare
import numpy.random as nprand
import numpy as np

# creates data and contingency tables
def generate_data(N):
    # random data: healed or not with particular drug
    data_drug1 = nprand.choice([1,0], size=int(N/2), p=[p_drug1, 1-p_drug1])
    data_drug2 = nprand.choice([1,0], size=int(N/2), p=[p_drug2, 1-p_drug2])

    healed1 = sum(data_drug1)
    not_healed1 = N/2 - healed1
    healed2 = sum(data_drug2)
    not_healed2 = N/2 - healed2

    # observed data:
    O = np.array([healed1, not_healed1, healed2, not_healed2])
    # probabilities:
    P = O / N
    # row and col sums:
    P1r = P[0] + P[1]
    P2r = P[2] + P[3]
    P1c = P[0] + P[2]
    P2c = P[1] + P[3]
    # expected data:
    E = N * np.array([P1r*P1c, P1r*P2c, P2r*P1c, P2r*P2c])

    return O,E

# return p-value after xi-test
def do_xi_test(O,E):
    xi, pval= chisquare(f_obs=O, f_exp=E)
    return pval

# calculates p-values repeatedly and returns median
def calc_pval(repeats, patients):
    pvals = np.zeros(repeats)
    N = patients
    for i in range(repeats):
        O,E = generate_data(N)
        pval = do_xi_test(O,E)
        pvals[i] = pval
        
    return np.median(pvals)

# probabilities of drugs to work
p_drug1 = 0.63 # global
p_drug2 = 0.68 # global

# initialisation of patients and p-value
N = 100
pval = 1

print("Searching for the smallest N to obey p-value rule < 0.05...")
while pval > 0.05:
    pval = calc_pval(repeats=500, patients=N)
    N+=50

else:
    print("The sufficient number of patients to distinguish between drugs is: {} with p-value: {}%".format(N, round(100*pval, 2)))

from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# number of matrices
N=2000

# returning symmetric matrix of dimension 'dim' and random values from distribution 'distro'
def generate_matrix(dim, distro):
    distributions = {
        "binar": nprand.choice([1,0], size=dim**2),
        "uniform": nprand.uniform(low=-1, high=1, size=dim**2),
        "binar-sqrt": nprand.choice([-np.sqrt(3), np.sqrt(3)], size=dim**2),
        "normal": nprand.normal(loc=0, scale=1, size=dim**2),
        "laplace": nprand.laplace(loc=0, scale=1/np.sqrt(2), size=dim**2)
        }

    values = distributions[distro]
    shape = (dim, dim)
    matrix = values.reshape(shape)
    symmetric = np.tril(matrix) + np.tril(matrix, -1).T
    
    return symmetric

# plots histogram for generated eigenvalues
def plot_eigens(dim, distro): 
    eigens = np.zeros(N*dim)
    for i in range(N):
        data = generate_matrix(dim, distro)
        eigens[i*dim:(i+1)*dim] = eigvals(data)
        
    #plt.hist(eigens, bins=20, alpha=1, edgecolor='black', linewidth=1.2, normed=True, label=str(dim))
    ax = sns.distplot(eigens, hist_kws=dict(edgecolor="k", linewidth=2), hist=False, label=str(dim))
    
# plots histograms for various matrix dimensions
def iterate_dim(min_dim, max_dim, step, distro):
    plt.figure(figsize=(15,5))
    print("Creating eigenvalue histogram for matrix of {} distribution..".format(str(distro)))
    for dim in range (min_dim,max_dim,step):
        plot_eigens(dim,distro)
    
    plt.legend(loc='upper right')
    plt.title(distro)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequencies")
    plt.show()

min_dim = 2
max_dim = 22
step = 2

iterate_dim(min_dim, max_dim, step, distro="binar")
iterate_dim(min_dim, max_dim, step, distro="uniform")
iterate_dim(min_dim, max_dim, step, distro="binar-sqrt")
iterate_dim(min_dim, max_dim, step, distro="normal")
iterate_dim(min_dim, max_dim, step, distro="laplace")



