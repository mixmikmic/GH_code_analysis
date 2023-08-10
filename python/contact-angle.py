from sympy import *
from sympy import init_printing
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import newton

init_printing(use_unicode=False, wrap_line=False, no_global=True)

def calculate_helmholtz_excess(
    rho_value,     
    reduced_temperature, 
    rho_inf, 
    eos
    ):
    # Defining symbols
    R, T, v, b, a, alpha, rho = symbols("R T v b a alpha rho")
    M, k = symbols("M xi")
    P_r, rho_r, T_r = symbols("P_r rho_r T_r")
    P_c, rho_c, T_c = symbols("P_c rho_c T_c")
    omega_a, omega_b = symbols("Omega_a Omega_b")
    Z_c = symbols("Z_c")
    rho_oo = symbols("rho_oo")
    
    P1 = eos.pressure.subs(v, M/rho)
    P2 = P1.subs(rho, rho_c * rho_r)
    P3 = P2.subs(T, T_c * T_r)
    P4 = P3/P_c
    P5 = P4.subs(a, omega_a*(R**2)*(T_c**2)/P_c)
    P6 = P5.subs(b, omega_b*R*T_c/P_c)
    P7 = expand(P6)
    P8 = P7.subs(T_c, P_c * M / (Z_c * rho_c * R)) 
    
    P_r = simplify(P8)    
    P_oo = P_r.subs(rho_r, rho_oo)    
    integrand = ((P_r - P_oo)/(rho_r ** 2))

    integrand = integrand.subs(omega_a, eos.omega_a)
    integrand = integrand.subs(omega_b, eos.omega_b)
    integrand = integrand.subs(Z_c, eos.critical_z_factor)
    integrand = integrand.subs(T_r, reduced_temperature)
    integrand = integrand.subs(rho_oo, rho_inf)
    
    func = lambdify(rho_r, integrand)
    y, err = integrate.quad(func, rho_inf, rho_value)
    return rho_value * y

# Defining symbols
P, R, T, v, b, a, alpha, rho = symbols("P R T v b a alpha rho")
def find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type):
    coef_a = (3.0 * p2 - (p1 ** 2)) / 3.0        
    coef_b = (2.0 * (p1 ** 3) - 9.0 * p1 * p2 + 27.0 * p3) / 27.0        
    delta = 0.25 * (coef_b ** 2) + (coef_a ** 3) / 27.0     

    if delta > 0.0:
        # 1 real root, 2 imaginary                 
        const_A =  cbrt(-0.5 * coef_b + sqrt(delta)) 
        const_B =  cbrt(-0.5 * coef_b - sqrt(delta))

        correct_root = const_A + const_B - p1 / 3.0 
    else:
        # 3 real roots
        phi = acos(-0.5 * coef_b / sqrt(-(coef_a ** 3) / 27.0))
        root_1 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0) - p1 / 3.0
        root_2 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 2.0 * np.pi / 3.0) - p1 / 3.0
        root_3 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 4.0 * np.pi / 3.0) - p1 / 3.0

        smallest_root = min(min(root_1,root_2), root_3)
        largest_root = max(max(root_1,root_2), root_3)

        if fluid_type is 'liquid':        
            correct_root = smallest_root
        else:
            assert fluid_type is 'vapor', 'Wrong fluid type! ' + fluid_type
            correct_root = largest_root
    
    assert correct_root > 0.0, fluid_type + ' Z-factor < 0.0! Delta is %f, %f' % (delta, correct_root)
    
    return correct_root

GAS_CONST = 8.314

class EquationOfState(object):
    def __init__(
        self,
        critical_pressure, 
        critical_temperature,
        acentric_factor,
        molar_mass
    ):       
        self.critical_pressure = critical_pressure
        self.critical_temperature = critical_temperature
        self.acentric_factor = acentric_factor
        self.molar_mass = molar_mass
        
        
    def calculate_pressure_and_densities(self, temperature):
        
        def func(x):            
            pressure = x
            Z_V = eos.calculate_eos_roots(pressure, temperature, 'vapor')
            Z_L = eos.calculate_eos_roots(pressure, temperature, 'liquid')
        
            f_V = eos.calculate_fugacity(pressure, temperature, Z_V)
            f_L = eos.calculate_fugacity(pressure, temperature, Z_L)
            
            return f_V - f_L
        
        pressure = newton(func=func, x0=1.0e5)
        
        Z_V = eos.calculate_eos_roots(pressure, temperature, 'vapor')
        Z_L = eos.calculate_eos_roots(pressure, temperature, 'liquid')
        
        rho_V = pressure * self.molar_mass / (Z_V * GAS_CONST * temperature)
        rho_L = pressure * self.molar_mass / (Z_L * GAS_CONST * temperature)
        
        return pressure, rho_V, rho_L
    
    
class PengRobinsonEos(EquationOfState):
    def __init__(
        self,
        critical_pressure, 
        critical_temperature,
        acentric_factor,
        molar_mass
    ):
        self.omega_a = 0.45724
        self.omega_b = 0.07780
        self.critical_z_factor = 0.3074
        self.pressure = R * T / (v - b) - a*alpha / (v*(v + b) + b*(v - b))
        
        EquationOfState.__init__(self, critical_pressure, critical_temperature, acentric_factor, molar_mass)

    
    def alpha_function(self, reduced_temperature):
        omega = self.acentric_factor
        
        if omega < 0.49:
            m = 0.37464 + 1.54226 * omega - 0.26992 * (omega ** 2)
        else:
            m = 0.379642 + 1.48503 * omega - 0.164423 * (omega ** 2) + 0.016667 * (omega ** 3)
        
        return (1.0 + m * (1.0 - sqrt(reduced_temperature))) ** 2
    
    
    def calculate_fugacity(
        self,
        pressure,
        temperature,
        z_factor
    ):
        P = pressure
        T = temperature
        Z = z_factor
        
        Pr = P / self.critical_pressure
        Tr = T / self.critical_temperature
        
        aa = self.alpha_function(Tr)
        
        A = aa * self.omega_a * Pr / (Tr ** 2)
        B = self.omega_b * Pr / Tr
        
        SQRT_2 = sqrt(2.0)        
        ln_f = (Z - 1.0) - log( Z - B ) - (A / (2.0 * SQRT_2 * B))              * log( (Z + (1.0 + SQRT_2)*B)/(Z + (1.0 - SQRT_2)*B) )
        return P * exp(ln_f)
        
        
    def calculate_eos_roots(self, pressure, temperature, fluid_type):
        P = pressure
        T = temperature

        Pr = P / self.critical_pressure
        Tr = T / self.critical_temperature
        
        aa = self.alpha_function(Tr)
        
        A = aa * self.omega_a * Pr / (Tr ** 2)
        B = self.omega_b * Pr / Tr      
        
        p0 = 1.0
        p1 = - (1.0 - B)
        p2 = A - 3.0 * (B ** 2) - 2.0 * B
        p3 = -(A * B - B ** 2 - B ** 3)     

        return find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type)       
        
        
class VanDerWaalsEos(EquationOfState):
    def __init__(
        self,
        critical_pressure, 
        critical_temperature,
        acentric_factor,
        molar_mass
    ):
        self.omega_a = 27.0/64.0
        self.omega_b = 1.0/8.0
        self.critical_z_factor = 3.0/8.0
        self.pressure = R * T / (v - b) - a / (v**2)
       
        EquationOfState.__init__(self, critical_pressure, critical_temperature, acentric_factor, molar_mass)
        
    def calculate_fugacity(
        self,
        pressure,
        temperature,
        z_factor
    ):
        P = pressure
        T = temperature
        Z = z_factor
        Pr = P / self.critical_pressure
        Tr = T / self.critical_temperature
        
        A = self.omega_a * Pr / (Tr ** 2)
        B = self.omega_b * Pr / Tr
        
        ln_f = B / (Z - B) - log(Z - B) - 2.0 * A / Z
        return P * exp(ln_f)
    
    
    def calculate_eos_roots(self, pressure, temperature, fluid_type):
        P = pressure
        T = temperature

        Pr = P / self.critical_pressure
        Tr = T / self.critical_temperature
        
        A = self.omega_a * Pr / (Tr ** 2)
        B = self.omega_b * Pr / Tr
        
        p0 = 1.0
        p1 = - (B + 1.0)
        p2 = A
        p3 = - A * B   
        
        return find_correct_root_of_cubic_eos(p0, p1, p2, p3, fluid_type) 

temperature = 303.15
pressure = 1.0 * 1.0e5

# n-Butane properties
critical_pressure = 6894.75729 * 550.7
critical_temperature = (5.0 / 9.0) * 765.3 # [K]
acentric_factor = 0.192800 # [-]
molar_mass = 58.12 * 0.001 # [kg/mol]

eos = VanDerWaalsEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)

pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)
Z_V = eos.calculate_eos_roots(pressure, temperature, 'vapor')
Z_L = eos.calculate_eos_roots(pressure, temperature, 'liquid')
f_V = eos.calculate_fugacity(pressure, temperature, Z_V)
f_L = eos.calculate_fugacity(pressure, temperature, Z_L)

assert abs(f_L - f_V) < 1.0e-6

# n-Butane properties
critical_pressure = 6894.75729 * 550.7
critical_temperature = (5.0 / 9.0) * 765.3 # [K]
acentric_factor = 0.192800 # [-]
molar_mass = 58.12 * 0.001 # [kg/mol]

reduced_temperatures = np.array([0.8, 0.82, 0.87, 0.9])

for reduced_temperature in reduced_temperatures:
    temperature = reduced_temperature * critical_temperature    
    
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)
    pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)

    rho_c = eos.critical_pressure * eos.molar_mass / (eos.critical_z_factor * GAS_CONST * eos.critical_temperature)
    rho_inf = rho_V / rho_c 
    
    eos.pressure = eos.pressure.subs(alpha, eos.alpha_function(reduced_temperature))    
    
    reduced_densities = np.linspace(0.1, 2.6, num=40)
    
    psi_vec = []
    reduced_densities_to_plot = []
    
    for rho in reduced_densities:
        psi = calculate_helmholtz_excess(
            rho_value=rho, 
            reduced_temperature=reduced_temperature, 
            rho_inf=rho_inf,
            eos=eos
            )

        reduced_densities_to_plot.append(rho)
        psi_vec.append(psi)
        
    psi_vec = np.array(psi_vec)
    plt.plot(reduced_densities_to_plot, psi_vec, marker='o', 
             label='Tr = %.2f' %reduced_temperature)

ax = plt.gca()
ax.grid(True)
ax.set_xlabel(xlabel='Reduced Density [-]')
ax.set_ylabel(ylabel=r'$\Psi^*_{exc}$')
ax.legend(loc='upper right')
plt.show()

def integrate_helholtz(eos, reduced_temperature, rho_0_r, rho_1_r, rho_inf):
    def func(x):
        psi = calculate_helmholtz_excess(
                rho_value=x, 
                reduced_temperature=reduced_temperature, 
                rho_inf=rho_inf,
                eos=eos
                )
        return psi
        
    y, err = integrate.quad(func, rho_0_r, rho_1_r)
    return sqrt(2.0 * y)

# Experimental surface tension n-butane
# T [K]   gamma [N/m]
# 230       0.02064
# 288       0.012
#reduced_temperature_points = np.array([230.0, 288]) / critical_temperature
#surface_tension_points = np.array([0.02064, 0.012])

# Experimental surface tension Water - H2O
reduced_temperature_points = np.array([0.422, 0.4297, 0.4374, 0.4529, 0.4683, 
                                       0.4838, 0.4992, 0.5147, 0.5301, 0.5456, 
                                       0.5610, 0.5765, 0.6537, 0.7310, 0.8082,
                                      0.8854, 0.9627, 0.9999])
surface_tension_experimental_points = np.array([0.0756, 0.0749, 0.0742, 0.0728, 0.0712,
                                   0.0696, 0.0679, 0.0662, 0.0644, 0.0626, 
                                   0.0608, 0.0589, 0.0482, 0.0376, 0.0264, 
                                   0.0147, 0.0037, 0.0])

# H2O properties
critical_pressure = 221.2 * 1.0e5
critical_temperature = 647.3 # [K]
acentric_factor = 0.344 # [-]
molar_mass = 18.0 * 0.001 # [kg/mol]

reduced_temperature_values = reduced_temperature_points[3:-2] # Two-phase values only
surface_tension_points = surface_tension_experimental_points[3:-2] # Two-phase values only

#reduced_temperature_values = reduced_temperature_points[3:4] # Two-phase values only
#surface_tension_points = surface_tension_experimental_points[3:4] # Two-phase values only

kappa_values = []

for reduced_temperature, surface_tension in zip(reduced_temperature_values, surface_tension_points):
    temperature = reduced_temperature * critical_temperature    
    
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)
    eos.pressure = eos.pressure.subs(alpha, eos.alpha_function(reduced_temperature))  
    #eos = VanDerWaalsEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)
    
    pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)

    rho_c = eos.critical_pressure * eos.molar_mass / (eos.critical_z_factor * GAS_CONST * eos.critical_temperature)
    #rho_c = 197.29
    rho_V_r = rho_V / rho_c
    rho_L_r = rho_L / rho_c
    
    integral = integrate_helholtz(eos, reduced_temperature, rho_V_r, rho_L_r, rho_inf=rho_V_r)

    kappa = ( surface_tension / (rho_c * sqrt(eos.critical_pressure) * integral) ) ** 2
    kappa_values.append(kappa)
    
kappa_values = np.array(kappa_values)    
plt.semilogy(surface_tension_points, kappa_values, marker='o', label='Peng-Robinson')

#plt.plot(gamma_calc, kappa_values, marker='o')
ax = plt.gca()
ax.grid(True, which=u'both')
ax.set_xlabel(xlabel='Surface Tension [N/m]')
ax.set_ylabel(ylabel=r'Surface parameter $\kappa$ ($N / \left( kg/m^3 \right)^2$)')
ax.legend(loc='upper left')

plt.show()

plt.semilogy(surface_tension_points, kappa_values, marker='o', label='Peng-Robinson')

ax = plt.gca()
ax.grid(True, which=u'both')
ax.set_xlabel(xlabel='Surface Tension [N/m]')
ax.set_ylabel(ylabel=r'Surface parameter $\kappa$ ($N / \left( kg/m^3 \right)^2$)')
ax.legend(loc='upper left')
ax.axis([0.01, 0.08, 1.0e-17, 1.0e-14])

plt.show()

sqrt_psi_vec = {}
reduced_densities_to_plot = {}

# n-Butane properties
critical_pressure = 6894.75729 * 550.7
critical_temperature = (5.0 / 9.0) * 765.3 # [K]
acentric_factor = 0.192800 # [-]
molar_mass = 58.12 * 0.001 # [kg/mol]

reduced_temperature = 0.8
temperature = reduced_temperature * critical_temperature

eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)
eos.pressure = eos.pressure.subs(alpha, eos.alpha_function(reduced_temperature))  
pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)
rho_c = eos.critical_pressure * eos.molar_mass / (eos.critical_z_factor * GAS_CONST * eos.critical_temperature)

rho_V_r = rho_V / rho_c
rho_L_r = rho_L / rho_c

reduced_densities = np.linspace(0.00001, 3.5, num=150)
    
sqrt_psi_vec[reduced_temperature] = []
reduced_densities_to_plot[reduced_temperature] = []

for rho in reduced_densities:
    psi = calculate_helmholtz_excess(
        rho_value=rho, 
        reduced_temperature=reduced_temperature, 
        rho_inf=rho_V_r,
        eos=eos
        )

    reduced_densities_to_plot[reduced_temperature].append(rho)
    sqrt_psi_vec[reduced_temperature].append(sqrt(2.0*psi))

sqrt_psi_vec[reduced_temperature] = np.array(sqrt_psi_vec[reduced_temperature])

plt.plot(reduced_densities_to_plot[reduced_temperature], 
         sqrt_psi_vec[reduced_temperature], marker='o', label='Reduced temperature = %.2f' %reduced_temperature)

surf_energy = 0.3
phi_r = surf_energy * np.ones(len(reduced_densities_to_plot[reduced_temperature]))
plt.plot(reduced_densities_to_plot[reduced_temperature], phi_r, 'k--', label='Specific Surface Energy = %.2f' %surf_energy)

ax = plt.gca()
ax.grid(True)
ax.set_xlabel(xlabel='Reduced density [-]')
ax.set_ylabel(ylabel=r'$\sqrt{2\Psi^*_{exc}}$')
ax.legend(loc='upper left')

plt.show()

def calculate_surface_densities(phi_r, reduced_temperature, rho_V_r, rho_L_r, rho_inf):
    def func(x):
        psi = calculate_helmholtz_excess(
            rho_value=x, 
            reduced_temperature=reduced_temperature, 
            rho_inf=rho_inf,
            eos=eos
            )
        
        if phi_r > 0.0:
            return phi_r - sqrt(2.0 * psi)
        else:
            return phi_r + sqrt(2.0 * psi)

    if phi_r > 0.0:
        rho_SV_r = newton(func=func, x0=rho_V_r + 0.1*rho_V_r)
        rho_SL_r = newton(func=func, x0=rho_L_r + 0.1*rho_L_r)
    else:       
        from scipy.optimize import brentq
        rho_SV_r = brentq(f=func, a=0.000001, b=0.9999*rho_V_r)
        rho_SL_r = brentq(f=func, a=0.5*(rho_L_r + rho_V_r), b=0.9999*rho_L_r)
    
    return rho_SV_r, rho_SL_r

def calculate_contact_angle(phi_r, gamma_VL_r, eos, reduced_temperature, rho_V_r, rho_L_r):
    if abs(phi_r) < 1.0e-8:
        contact_angle = 90.0
    else:
        rho_SV_r, rho_SL_r = calculate_surface_densities(
            phi_r=phi_r, 
            reduced_temperature=reduced_temperature, 
            rho_V_r=rho_V_r, 
            rho_L_r=rho_L_r, 
            rho_inf=rho_V_r
        )

        if phi_r > 0.0:
            integral_SV = integrate_helholtz(eos, reduced_temperature, rho_V_r, rho_SV_r, rho_inf=rho_V_r)
            integral_SL = integrate_helholtz(eos, reduced_temperature, rho_L_r, rho_SL_r, rho_inf=rho_V_r)
        else:
            integral_SV = integrate_helholtz(eos, reduced_temperature, rho_SV_r, rho_V_r, rho_inf=rho_V_r)
            integral_SL = integrate_helholtz(eos, reduced_temperature, rho_SL_r, rho_L_r, rho_inf=rho_V_r)    

        gamma_SV_r = integral_SV - phi_r * rho_SV_r
        gamma_SL_r = integral_SL - phi_r * rho_SL_r

        contact_angle = acos((gamma_SV_r - gamma_SL_r) / gamma_VL_r) * 180.0 / np.pi
    
    return contact_angle

contact_angles = {}
phi_values = {}
phi_limits = {}
# Phi limits (so functions do not diverge)
# Tr = 0.9 -> phi_r = 0.35
# Tr = 0.87 -> phi_r = 0.43
# Tr = 0.82 -> phi_r = 0.565
# Tr = 0.8 -> phi_r = 0.615
phi_limits[0.9] = 0.35
phi_limits[0.87] = 0.43
phi_limits[0.82] = 0.565
phi_limits[0.8] = 0.615

# n-Butane properties
critical_pressure = 6894.75729 * 550.7
critical_temperature = (5.0 / 9.0) * 765.3 # [K]
acentric_factor = 0.192800 # [-]
molar_mass = 58.12 * 0.001 # [kg/mol]


reduced_temperature_values = [0.8, 0.82, 0.87, 0.9]
for reduced_temperature in reduced_temperature_values:
    temperature = reduced_temperature * critical_temperature
    eos = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor, molar_mass)
    rho_c = eos.critical_pressure * eos.molar_mass / (eos.critical_z_factor * GAS_CONST * eos.critical_temperature)

    pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)
    eos.pressure = eos.pressure.subs(alpha, eos.alpha_function(reduced_temperature))

    pressure, rho_V, rho_L = eos.calculate_pressure_and_densities(temperature)

    rho_V_r = rho_V / rho_c
    rho_L_r = rho_L / rho_c

    gamma_VL_r = integral_VL = integrate_helholtz(eos, reduced_temperature, rho_V_r, rho_L_r, rho_inf=rho_V_r)

    phi_values[reduced_temperature] = np.linspace(-phi_limits[reduced_temperature], phi_limits[reduced_temperature], num=25)

    contact_angles[reduced_temperature] = []

    for phi_r in phi_values[reduced_temperature]:
        contact_angle = calculate_contact_angle(
            phi_r, 
            gamma_VL_r=gamma_VL_r, 
            eos=eos, 
            reduced_temperature=reduced_temperature,
            rho_V_r=rho_V_r,
            rho_L_r=rho_L_r
        )
        contact_angles[reduced_temperature].append(contact_angle)

    contact_angles[reduced_temperature] = np.array(contact_angles[reduced_temperature])

    plt.plot(phi_values[reduced_temperature], 
             contact_angles[reduced_temperature], 
             marker='o', label='Reduced temperature = %.2f' %reduced_temperature)

ax = plt.gca()
ax.grid(True)

ax.set_xlabel(xlabel='Specific Surface Energy [-]')
ax.set_ylabel(ylabel='Contact Angle [deg]')
ax.legend(loc='upper right')

plt.show()

# Only Plot
reduced_temperature_values = [0.8, 0.82, 0.87, 0.9]

for reduced_temperature in reduced_temperature_values:
    plt.plot(phi_values[reduced_temperature], 
             contact_angles[reduced_temperature], 
             marker='o', label='Reduced temperature = %.2f' %reduced_temperature)

ax = plt.gca()
ax.grid(True)

ax.set_xlabel(xlabel='Specific Surface Energy [-]')
ax.set_ylabel(ylabel='Contact Angle [deg]')
ax.legend(loc='upper right')

plt.show()

