import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

import models
import payoffs
import plotting
import selection_functions

get_ipython().magic('matplotlib inline')
plt.style.use("ggplot")

get_ipython().magic('pinfo integrate.solve_ivp')

# random initial condition
prng = np.random.RandomState(42)
number_of_genotypes = 4
initial_offspring_share, = prng.dirichlet(np.ones(number_of_genotypes), 1)
y0 = initial_offspring_share

# define the selection functions
d1, d3 = 2, 1
UGA = lambda x_A: 1
UgA = lambda x_A: selection_functions.kirkpatrick_selection(x_A, d3)

# define the payoffs, metabolic costs and mutation rate
payoffs.prisoners_dilemma_payoffs(prng)
M, m = 0.0, 0.0
metabolic_costs = np.array([M, m])
mutation_rate = 0.0

payoff_kernel

solution = integrate.solve_ivp(F, t_span=(0, 100), y0=y0, method="RK45", rtol=1e-12, atol=1e-15,
                               dense_output=True, vectorized=True)

solution

plt.plot(solution.t, solution.y[0], label="GA")
plt.plot(solution.t, solution.y[1], label="Ga")
plt.plot(solution.t, solution.y[2], label="gA")
plt.plot(solution.t, solution.y[3], label="ga")
plt.ylim(0, 1.05)
plt.legend()
plt.show()

# sliders used to control the initial condition
x1_slider = widgets.FloatSlider(value=0.25, min=0.0, max=1.0, step=1e-3, description=r"$x_1$", readout_format=".3f")
x2_slider = widgets.FloatSlider(value=0.25, min=0.0, max=1.0, step=1e-3, description=r"$x_2$", readout_format=".3f")
x3_slider = widgets.FloatSlider(value=0.25, min=0.0, max=1.0, step=1e-3, description=r"$x_3$", readout_format=".3f")

# sliders used to control the Prisoner's Dilemma Payoffs
T_slider = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description=r"$T$")
R_slider = widgets.FloatSlider(value=8, min=0, max=100, step=0.1, description=r"$R$")
P_slider = widgets.FloatSlider(value=6, min=0, max=100, step=0.1, description=r"$P$")
S_slider = widgets.FloatSlider(value=4, min=0, max=100, step=0.1, description=r"$S$")

# sliders used to control the metabolic costs
M_slider = widgets.FloatSlider(value=0, min=0, max=100, step=0.1, description=r"$M_G$")
m_slider = widgets.FloatSlider(value=0, min=0, max=100, step=0.1, description=r"$m_g$")

# slider used to control which selection function is being used
U_slider = widgets.Dropdown(options=["kirkpatrick", "seger", "wright"], index=0, description=r"$U_{\gamma(j)A}$")

# slider that controls the parameters of the selection function
d1_slider = widgets.FloatSlider(value=1, min=0.0, max=10, step=0.05, description=r"$d_1$")
d3_slider = widgets.FloatSlider(value=1, min=0.0, max=10, step=0.05, description=r"$d_3$")

# slider used to control the mutation rate
e_slider = widgets.FloatSlider(value=0.0, min=0.0, max=1.0, step=1e-3, description=r"$\epsilon$", readout_format=".3f")

# slider that controls max simulation time
max_time_slider = widgets.IntSlider(value=25, min=1, max=100000, description=r"$\max t$")

# slider used to control which selection function is being used
U_slider = widgets.Dropdown(options=["kirkpatrick", "seger", "wright"], index=0, description=r"$U_{\gamma(j)A}$")


w = widgets.interactive(plotting.plot_generalized_sexual_selection, x1=x1_slider, x2=x2_slider, x3=x3_slider,
                        selection_function=U_slider, d1=d1_slider, d3=d3_slider, 
                        T=T_slider, R=R_slider, P=P_slider, S=S_slider,
                        M=M_slider, m=m_slider, epsilon=e_slider,
                        max_time=max_time_slider)
display(w)

# can get access to the solution!
(solution, optimize_result) = w.result



