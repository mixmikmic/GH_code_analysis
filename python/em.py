get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# %cd .. \nimport sys\nsys.path.append("..")\nimport statnlpbook.util as util\nimport statnlpbook.em as em\nimport matplotlib.pyplot as plt\nimport mpld3\nimport numpy as np')

from math import log, exp

# Domains and values
z_domain = ['c1','c2']
x_domain = ['nattoo','pizza','fries']
c1, c2 = z_domain
n, p, f = x_domain

def prob(x, z, theta):
    """
    Calculate probability of p_\theta(x,z).
    Args:
        x: list of words of the document, should be in `x_domain`
        z: class label, should be in `z_domain`
    Returns:
        probability p(x,z) given the parameters.
    """
    theta_x, theta_z = theta
    bias = log(theta_z[z])
    ll = sum([log(theta_x[x_i, z]) for x_i in x])
    return exp(bias + ll)

def create_theta(prob_c1, prob_c2, 
                 n_c1, p_c1, f_c1, 
                 n_c2, p_c2, f_c2):
    """
    Construct a theta parameter vector. 
    """
    theta_z = { c1: prob_c1, c2: prob_c2}
    theta_x = { (n, c1): n_c1, (p, c1): p_c1, (f, c1): f_c1, 
                (n, c2): n_c2, (p, c2): p_c2, (f, c2): f_c2}
    return theta_x, theta_z
    

theta = create_theta(0.5,0.5, 0.3, 0.5, 0.2, 0.1, 0.4, 0.5)
prob([p,p,f], c2, theta)

def marginal_ll(data, theta):
    """
    Calculate the marginal log-likelihood of the given `data` using parameter `theta`.
    Args:
        data: list of documents, where each document is a list of words. 
        theta: parameters to use.  
    """
    return sum([log(prob(x,c1,theta) + prob(x,c2,theta)) for x in data]) / len(data)

marginal_ll([[p,p,f],[n,n]], theta)

theta1 = create_theta(0.3, 0.7, 0.0, 0.3, 0.7, 1.0, 0.0, 0.0)
theta2 = create_theta(0.3, 0.7, 1.0, 0.0, 0.0, 0.0, 0.3, 0.7)

em.plot_1D(lambda theta: marginal_ll(dataset, theta), theta1, theta2, ylim=[-8.5,-5.5])

current_theta = add_theta(0.4, theta1, 0.6, theta2)

def calculate_class_distributions(data, theta):
    result = []
    for x in data:
        norm = prob(x,c1,theta) + prob(x,c2,theta)
        # E Step
        q = {
            c1: prob(x,c1,theta) / norm,
            c2: prob(x,c2,theta) / norm
        }
        result.append(q)
    return result

current_q = calculate_class_distributions(dataset, current_theta)

def marginal_ll_bound(data, theta, q_data = current_q):
    loss = 0.0
    for x,q in zip(data,q_data):
        loss += q[c1] * log(prob(x,c1,theta) / q[c1]) + q[c2] * log(prob(x,c2,theta) / q[c2])
    return loss / len(data)

em.plot_1D(lambda theta: marginal_ll(dataset, theta), theta1, theta2, 
           loss2=lambda theta:marginal_ll_bound(dataset,theta), ylim=[-8.5,-5.5])

from collections import defaultdict

def e_step(data,theta):
    return calculate_class_distributions(data, theta)

def m_step(x_data,q_data):
    counts = defaultdict(float)
    norm = defaultdict(float)
    class_counts = defaultdict(float)
    for x,q in zip(x_data, q_data):
        for z in z_domain:
            class_counts[z] += q[z]
            for x_i in x:
                norm[z] += q[z]
                counts[x_i, z] += q[z]
    theta_c = dict([(z,class_counts[z] / len(x_data)) for z in z_domain])
    theta_x = dict([((x,z),counts[x,z] / norm[z]) for z in z_domain for x in x_domain])
    return theta_x, theta_c

def em_algorithm(data, init_theta, iterations = 10):
    current_theta = init_theta
    current_q = None
    result = []
    for _ in range(0, iterations):
        current_q = e_step(data, current_theta)
        current_theta = m_step(data, current_q)
        current_marg_ll = marginal_ll(data, current_theta)
        current_bound = marginal_ll_bound(data, current_theta, current_q)
        result.append((current_q, current_theta, current_marg_ll, current_bound))
    return result

data = [[p,p,p,p,p,n],[n,n,n,n,n,n,f,p],[f,f,f,f,p,p,p,n]]
iterations = em_algorithm(data, current_theta, 5)
iterations[-1]

fig = plt.figure()
plt.plot(range(0, len(iterations)), [iteration[2] for iteration in iterations], label='marg_ll')
plt.plot(range(0, len(iterations)), [iteration[3] for iteration in iterations], label='bound')
plt.legend()
mpld3.display(fig)

