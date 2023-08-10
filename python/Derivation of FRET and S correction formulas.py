import sympy
from sympy import init_printing, symbols, solve, Eq
init_printing()  # beautiful maths

E, Er, Epr = symbols('E, E_R, E_PR')
S, Sr, Spr = symbols('S, S_R, S_PR')
nd, na, ns_a, n_dir, naa, nt = symbols('n_d n_a n^*_a n_dir n_aa n_t')
gamma, Lk = symbols('gamma L_k')
d_exAA, d_exT, d_exD = symbols("d_dirAA d_dirT, d_dirD")

def PR(nd, na):
    return na/(na + nd)

PR(nd, na)

def FRET(nd, na, gamma):
    return na / (na + gamma*nd)

FRET(nd, na, gamma)

ns_a_as_func_na = na + Lk*nd + n_dir
ns_a_as_func_na

Er_sym = sympy.factor(PR(nd, ns_a).subs(ns_a, ns_a_as_func_na).subs(n_dir, d_exT*(nd*gamma + na)))
Er_sym

E_func_Er = sympy.factor(FRET(nd, na, gamma).subs(nd, solve(Er_sym - Er, nd)[0])).collect(Er)
E_func_Er

print(sympy.printing.latex(E_func_Er))

E_func_Er

print(E_func_Er)

def correct_E_gamma_leak_dir(E_R, gamma, L_k=0, d_dirT=0):
    E_R = np.asarray(E_R)
    return ((E_R*(L_k + d_dirT*gamma + 1) - L_k - d_dirT*gamma) / 
            (E_R*(L_k - gamma + 1) - L_k + gamma))

E_func_Er.subs(Lk, 0).subs(d_exT, 0)

sympy.collect(E_func_Er.subs(gamma, 1).subs(d_exT, 0), Er)

sympy.collect(E_func_Er.subs(gamma, 1).subs(Lk, 0), Er)

Er_func_E = solve(E - E_func_Er, Er)[0]

sympy.collect(Er_func_E, E)

print(sympy.collect(Er_func_E, E))

def uncorrect_E_gamma_leak_dir(E, gamma, L_k=0, d_dirT=0):
    E = np.asarray(E)
    return ((E*(-L_k + gamma) + L_k + d_dirT*gamma) /
            (E*(-L_k + gamma - 1) + L_k + d_dirT*gamma + 1))

Er_func_E.subs(Lk, 0).subs(d_exT, 0)

sympy.collect(Er_func_E.subs(gamma, 1).subs(d_exT, 0), Er)

sympy.collect(Er_func_E.subs(gamma, 1).subs(Lk, 0), Er)

def StoichRaw(nd, na, naa):
    return (na + nd)/(na + nd + naa)

StoichRaw(nd, na, naa)

def Stoich(nd, na, naa, gamma):
    return (na + gamma*nd)/(na + gamma*nd + naa)

Stoich(nd, na, naa, gamma)

ns_a_as_func_na = na + Lk*nd + n_dir
ns_a_as_func_na

Sr_sym = sympy.factor(
    StoichRaw(nd, ns_a, naa).subs(ns_a, ns_a_as_func_na).subs(n_dir, d_exT*(na + gamma*nd))
    )
Sr_sym

S_sym = Stoich(nd, na, naa, gamma)
S_sym

S_func_Sr_nx = S_sym.subs(na, solve(Sr_sym - Sr, na)[0]).factor()
S_func_Sr_nx

E_sym = FRET(nd, na, gamma)
E_sym

na_func_Sr = solve(Sr_sym - Sr, na)[0]
na_func_Sr

E_func_Sr = E_sym.subs(na, na_func_Sr).factor()
E_func_Sr

nd_func_E_Sr = solve(E_func_Sr - E, nd)[0]
nd_func_E_Sr

S_func_E_Sr = S_func_Sr_nx.replace(nd, nd_func_E_Sr).factor()
S_func_E_Sr

ns_a_as_func_na

Er_sym = PR(nd, ns_a).subs(ns_a, ns_a_as_func_na).subs(n_dir, d_exT*(nd*gamma + na))
Er_sym

E_func_Er = FRET(nd, na, gamma).subs(nd, solve(Er_sym - Er, nd)[0]).factor()
E_func_Er

S_func_Er_Sr = S_func_E_Sr.replace(E, E_func_Er).factor()
S_func_Er_Sr 

print(sympy.printing.latex(S_func_Er_Sr))

print(S_func_Er_Sr)

def correct_S(E_R, S_R, gamma, L_k, d_dirT):
    return (S_R*(E_R*L_k - E_R*gamma + E_R - L_k + gamma) /
            (E_R*L_k*S_R - E_R*S_R*gamma + E_R*S_R - L_k*S_R - S_R*d_dirT + 
             S_R*gamma - S_R + d_dirT + 1))

S_func_E_Spr = (S_func_E_Sr.replace(Lk, 0).replace(d_exT, 0)
                .replace(Sr, Spr).replace(Er, Epr))
S_func_E_Spr

S_func_Epr_Spr = (S_func_Er_Sr.replace(Lk, 0).replace(d_exT, 0)
                  .replace(Sr, Spr).replace(Er, Epr))
S_func_Epr_Spr

S_func_Er_Sr

Sr_func_Er_S = solve(S_func_Er_Sr - S, Sr)[0]
Sr_func_Er_S

print(Sr_func_Er_S)

def uncorrect_S(E_R, S, gamma, L_k, d_dirT):
    return (S*(d_dirT + 1) / 
            (-E_R*L_k*S + E_R*L_k + E_R*S*gamma - E_R*S - E_R*gamma + 
             E_R + L_k*S - L_k + S*d_dirT - S*gamma + S + gamma))

import numpy as np

Ex = np.arange(-0.2, 1.2, 0.1)

gamma_ = 0.75
leakage_ = 0.04
dir_ex_t_ = 0.08

Ex_roundtrip = uncorrect_E_gamma_leak_dir(correct_E_gamma_leak_dir(Ex, gamma_), gamma_)

np.allclose(Ex, Ex_roundtrip)

Ex_roundtrip = uncorrect_E_gamma_leak_dir(correct_E_gamma_leak_dir(Ex, gamma_, leakage_, dir_ex_t_), 
                                          gamma_, leakage_, dir_ex_t_)

np.allclose(Ex, Ex_roundtrip)

Ex_roundtrip = correct_E_gamma_leak_dir(uncorrect_E_gamma_leak_dir(Ex, gamma_, leakage_, dir_ex_t_), 
                                        gamma_, leakage_, dir_ex_t_)

np.allclose(Ex, Ex_roundtrip)

Ex = np.arange(-0.2, 1.2, 0.01)
Sx = np.arange(-0.2, 1.2, 0.01)

np.random.shuffle(Ex)
np.random.shuffle(Sx)

gamma_ = 1
leakage_ = 0
dir_ex_t_ = 0

S_corr = correct_S(Ex, Sx, gamma_, leakage_, dir_ex_t_)
S_uncorr = uncorrect_S(Ex, S_corr, gamma_, leakage_, dir_ex_t_)

np.allclose(S_corr, Sx)

np.allclose(S_uncorr, Sx)

gamma_ = 0.7
leakage_ = 0.05
dir_ex_t_ = 0.1

S_corr = correct_S(Ex, Sx, gamma_, leakage_, dir_ex_t_)
S_uncorr = uncorrect_S(Ex, S_corr, gamma_, leakage_, dir_ex_t_)

np.allclose(S_uncorr, Sx)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

idx = Sx.argsort()

plt.plot(S_corr[idx], 'o')



