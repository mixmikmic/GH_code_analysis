import math as m

# propellant properties and physical constants

rho = 789                             # propellant density (ethanol and LOX respectively) [kg/m^3]
p_v = 8.84E3                          # propellant vapor pressure [Pa]
g_0 = 9.81                            # gravitational acceleration [m/s/s]
 
# rocket model (assuming sea-level operation)

isp = 246 * 0.90                          # specific impulse (sea-level, optimistic estimate) [s]
f = 4.5E3                                 # thrust [N]
mdot_t = f/(g_0 * isp)                    # total mass flowrate [kg/s]
chamber_p= 380 * 6.895 * 1000             # chamber pressure (assumed to be 380 psi currently) [Pa] 
loss_factor = 1.15                        # estimate of line and injector losses
OF = 1.3                                  # mixture ratio
mdot_o = mdot_t / (1 + (1/OF))            # oxidizer mass flowrate [kg/s]
mdot_f = mdot_t / (1 + OF)                # fuel mass flowrate [kg/s]
p_i = 101.3E3                             # inlet pressure (currently 1 atm) [Pa]
delta_p = chamber_p * loss_factor - p_i   # required pump discharge pressure [Pa]

#barleycorn conversions
g_0 = 32.2
delta_p = delta_p * 0.000145038
mdot = mdot_f * 2.20462
rho = rho * 0.062428
h_i = p_i * 0.000145038 * 144 / rho
h_v = p_v * 0.000145038 * 144 / rho

# derived parameters

q = mdot / rho                      # volumetric flowrate [f^3/s]
h_p = 144 * delta_p / rho           # required head rise [ft]
npsh_a = h_i - h_v    # Net Positive Suction Head (available) [ft]

print("mass flow rate")
print("%.3f" % mdot, "lbm/s ", "\n")
print("volumetric flow rate")
print("%.4f" % q, "ft^3/s", "\n")
print("required pressure head")
print("%.3f" % h_p, "ft", "\n")
print("Net Positive Suction Head Available")
print("%.3f" % npsh_a, "ft")

# number of stages

delta_p_s = 47E6 * 0.000145038         # estimated allowable pressure rise per stage [Pa]
n = int(m.ceil(delta_p/delta_p_s))     # number of stages

print("number of stages")
print(n)

from IPython.display import Image
Image(filename='specific_speed.png') 

# pump rotational speed

psi = 1                                         # pump stage head coefficient (estimated)
u_t = psi * m.sqrt(2 * g_0 * h_p)               # impeller vane tip speed [ft/s]
u_ss = 10E3                                     # suction specific speed
npsh_r = npsh_a * 0.8
n_r = (u_ss * npsh_r**0.75)/(21.2 * m.sqrt(q))  # pump rotational speed [RPM]
n_r_rad = n_r * 2 * m.pi / 60
n_s = (21.2* n_r * m.sqrt(q))/(h_p / n)**0.75   # pump specific speed

print("impeller tip speed")
print("%.2f" % u_t, "ft/s", "\n")
print("pump rotational speed")
print("%.2f" % n_r, "RPM", "\n")
print("pump specific speed")
print("%.2f" % n_s)

# impeller diameters

l = 0.3                                                        # hub to tip diameter ratio (assume 0.3)
phi = 1                                                        # inducer flow coefficicent (assume 0.1 with inducer, 1.0 otherwise)
d_o = u_t * 2 / n_r_rad * 12                                   # impeller outer diameter [in]
d_i = (4 * q/(m.pi * phi * n_r_rad * (1 - l**2)))**(1/3) * 12  # impeller inner diameter [in]

print("impeller discharge diameter")
print("%.3f" % d_o, "in", "\n")
print("impeller inlet diameter")
print("%.4f" % d_i, "in")

# shaft power

from numpy import interp

#from a table of efficiencies for radial impellers found in Munson et. al., 2009 
def effic(n_s):
    eta_t = [0.55, 0.80, 0.83, 0.85, 0.84, 0.82]
    n_s_t = [500, 1000, 1500, 2000, 2500, 3000]
    eta = interp(n_s, n_s_t, eta_t)
    return eta

eff = 0.4 # a more realistic estimated efficiency
#eta = input('given an pump specific speed of {} and flow rate of {} m^3/s what is the pump efficiency (from a lookup table in Munson) '.format("%.3f" % n_s, "%.3f" % q))
def power(mdot, h_p, n_s):  # required shaft power [W]
    p_req = mdot * h_p/(effic(n_s) * 0.738)
    return p_req

print("pump efficiency")
print("%.2f" % float(100*effic(n_s)), "%", "\n")
print("required shaft power (optimistic)")
print("%.2f" % power(mdot, h_p, n_s), "W", "\n")
print("required shaft power (realistic)")
print("%.2f" % power(mdot,h_p, eff), "W")

f_t = 50                                              #engine run time [s]
print("%.2f" % float(power(mdot,h_p, eff)*f_t/1000), "kJ")  #Total stored energy [kJ]
#changed for illustrative purposes

