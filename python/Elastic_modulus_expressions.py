from IPython.display import Image
Image("data/moduli.png")

import sympy
sympy.__version__

sympy.init_printing(use_latex='mathjax')

alpha, beta, gamma = sympy.symbols("alpha, beta, gamma")
lamda, mu, E, K, M, rho = sympy.symbols("lamda, mu, E, K, M, rho")
nu = sympy.symbols("nu")

from sympy import sqrt
alpha_expr = sqrt((mu * (E - 4*mu)) / (rho * (E - 3*mu)))
alpha_expr

print(sympy.latex(alpha_expr))

mu_expr = (3 * K * E) / (9 * K - E)

subs = alpha_expr.subs(mu, mu_expr)
subs

print(sympy.latex(subs))

from sympy import simplify
simplify(subs)

print(sympy.latex(simplify(subs)))

beta_expr = sqrt(mu/rho)
simplify(beta_expr.subs(mu, mu_expr))

simpl = simplify(beta_expr.subs(mu, mu_expr))

print(sympy.latex(simpl))

gamma_expr = sqrt((K + (4*mu/3)) / mu)
simpl = simplify(gamma_expr.subs(mu, mu_expr))
simpl

print(sympy.latex(simpl))

gamma_emu_expr = alpha_expr / beta_expr
simpl = sqrt(gamma_emu_expr**2)
simpl

print(sympy.latex(simpl))

e_expr = 2 * mu * (1 + nu)

simplify(sqrt(gamma_emu_expr.subs(E, e_expr)**2))

print(sympy.latex(simplify(sqrt(gamma_emu_expr.subs(E, e_expr)**2))))

vp_expr = sympy.Eq(alpha, sqrt(mu*(E - 4*mu) / (rho*(E - 3*mu))))
vp_expr

alpha_expr = sqrt(mu*(E - 4*mu) / (rho*(E - 3*mu)))

mu_expr = (rho * alpha**2 - lamda)/2

new_expr = simplify(vp_expr.subs(mu, mu_expr))

new_expr.subs(alpha, alpha_expr)

vp_wolfram = simplify(sqrt(-lamda/rho+sqrt(9*lamda**2+E**2+2*lamda*E)/rho+E/rho)/sqrt(2))**2
sqrt(vp_wolfram)

print(sympy.latex(sqrt(vp_wolfram)))

simplify(sqrt(sqrt(9*lamda**2+2*lamda*E+E**2)/rho-(3*lamda)/rho+E/rho)/2)**2

simplify(sqrt(sqrt(9*lamda**2+2*lamda*E+E**2)/E+(3*lamda)/E+3)/sqrt(2))**2

mu_expr = 3*K*(1 - 2*nu)/(2 + 2*nu)
simpl = simplify((sqrt(mu/rho)).subs(mu,mu_expr)**2)
simpl

print(sympy.latex(simpl))

vp_expr = sqrt(lamda * (1 - nu) / (rho * nu))
l_expr = 3 * K * nu / (1 + nu)
vp_expr.subs(lamda, l_expr)



