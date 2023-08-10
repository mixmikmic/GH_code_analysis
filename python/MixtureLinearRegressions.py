import util
from util import *
import models
from models.LinearRegressions import LinearRegressionsMixture
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from IPython.display import display, Markdown, Math
from operator import mul
import sympy as sp
sp.init_printing()

K, D = 2, 2
np.random.seed(0)
pis = np.array([0.4, 0.6])
ws = np.array([[0.75, 0.25], [0.4, 0.9]])
model = LinearRegressionsMixture.generate(K, D, betas = ws, weights = pis, cov =  1.0)
print("True parameters:")
print(model.weights)
print(model.betas)
print(model.sigma)

def evaluate_mixture(ws, pis, beta):
    """
    Compute E_\pi[w^beta]
    """
    return sum(pi * util.monomial(w, beta) for w, pi in zip(ws.T, pis))
    
def compute_exact_y_moments(ws, pis, moments_x, alpha, b):
    """
    Compute the exact moments E[x^a y^b] using coefficients ws, pis and moments_x.
    """
    D, K = ws.shape
    ret = 0.
    coeffs = sp.ntheory.multinomial_coefficients(D,b)
    for beta in partitions(D, b):
        ret += coeffs[beta] * evaluate_mixture(ws, pis, beta) * moments_x[tuple_add(alpha, beta)]
    return ret

def describe_moment_polynomial(R, moments_x, moment_y, alpha, b):
    """
    Computes the moment polynomial for E[x^alpha, y^b]
    """
    D = len(R.symbols)
    w = R.symbols
    expr = -moment_y
    coeffs = sp.ntheory.multinomial_coefficients(D,b)
    for beta in partitions(D, b):
        expr += coeffs[beta] * util.monomial(w, beta) * moments_x[tuple_add(alpha, beta)]
    return expr

# Example 
def double_factorial(n): 
    return reduce(mul, xrange(n, 0, -2)) if n > 0 else 1
def gaussian_moments(sigma, d):
    """
    E[x^d] where x is standard gaussian with sigma
    """
    if d == 0: return 1
    elif d % 2 == 0: return double_factorial(d-1) * sigma**d
    else: return 0
def expected_gaussian_moments(sigma, alphas):
    return {alpha : prod(gaussian_moments(sigma, a) for a in alpha) for alpha in alphas}
def expected_uniform_moments(alphas):
    return {alpha : 1. for alpha in alphas}
def expected_moments_y(ws, pis, moments_x, alphabs):
    return {(alpha, b) : compute_exact_y_moments(ws, pis, moments_x, alpha, b) for alpha, b in alphabs}

R, _ = sp.xring(['w%d'%d for d in xrange(D)], sp.RR, sp.grevlex)

deg_b, deg_x = 3, 2
sigma = model.sigma[0,0]
alphas = list(dominated_elements((deg_x for _ in xrange(D))))
alphabs = [(alpha, b) for alpha in alphas for b in xrange(1,deg_b+1)]
alphas_ = list(dominated_elements((deg_x + deg_b for _ in xrange(D))))
moments_x = expected_gaussian_moments(sigma, alphas_)
moments_y = expected_moments_y(ws, pis, moments_x, alphabs)
#display(moments)

display(moments_y)

def get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b):
    constrs = []
    for b in xrange(1, deg_b+1):
        for alpha in util.dominated_elements((deg_x for _ in xrange(D))):
            constrs.append( describe_moment_polynomial(R, moments_x, moments_y[(alpha, b)], alpha, b) )
    return constrs

from mompy.core import MomentMatrix
import mompy.solvers as solvers; reload(solvers)
import mompy.extractors as extractors; reload(extractors)

constrs = get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b)
M = MomentMatrix(3, R.symbols, morder='grevlex')
sol = solvers.solve_generalized_mom_coneqp(M, constrs, 3)
display( model.betas)
display(extractors.extract_solutions_lasserre(M, sol['x'], Kmax=2))

display(constrs)

def compute_expected_moments(xs, alphas):
    moments = {alpha : 0. for alpha in alphas}
    for alpha in alphas:
        m = monomial(xs.T, alpha)
        moments[alpha] = m if isinstance(m,float) else float(m.mean())
    return moments

def compute_expected_moments_y(ys, xs, alphabs):
    moments = {(alpha, b) : 0. for alpha, b in alphabs}
    for alpha, b in alphabs:
        moments[(alpha,b)] = float((monomial(xs.T, alpha) * ys**b).mean())
    return moments



from mompy.core import MomentMatrix
import mompy.solvers as solvers; reload(solvers)
import mompy.extractors as extractors; reload(extractors)
from util import fix_parameters, column_rerr
totalerror = 0
for t in range(1):
    xs, ys = model.sample(1e6)
    moments_x = compute_expected_moments(xs, alphas_)
    moments_y = compute_expected_moments_y(ys, xs, alphabs)

    constrs_sdp = get_constraint_polynomials(moments_y, moments_x, deg_x, deg_b)
    M = MomentMatrix(3, R.symbols, morder='grevlex')
    solsdp = solvers.solve_generalized_mom_coneqp(M, constrs_sdp, None)
    #sol = solvers.solve_moments_with_convexiterations(M, constrs, 3)
     
    betas_lassare = extractors.extract_solutions_lasserre(M, solsdp['x'], Kmax=2)
    betas_lassare_array = np.array([betas_lassare[R.symbols[0]], betas_lassare[R.symbols[1]]])
        
    M_ = closest_permuted_matrix(model.betas.T, betas_lassare_array.T)
    
    display(M_)
    display('the true parameters')
    display(model.betas.T)
    totalerror += column_aerr(M_, model.betas)
print totalerror
    



