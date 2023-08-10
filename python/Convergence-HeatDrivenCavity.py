get_ipython().magic('matplotlib inline')

import fenics

import phaseflow

def verify_against_wang2010(solution):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5,
        'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.],
        'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])

        values = solution(p)
        
        ux = values[0]*data['Pr']/data['Ra']**0.5
        
        assert(abs(ux - true_ux) < 2.e-2)
        
    print("Successfully verified against wang2010.")

def heat_driven_cavity(output_dir = "output/heat_driven_cavity",
        grid_size = 40,
        time_step_size = 0.001,
        end_time = 1.,
        stop_when_steady = True,
        steady_relative_tolerance = 1.e-4):

    def m_B(T, Ra, Pr, Re):

        return T*Ra/(Pr*Re**2)


    def ddT_m_B(T, Ra, Pr, Re):

        return Ra/(Pr*Re**2)
    
    
    solution, mesh = phaseflow.run(output_dir = output_dir,
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        thermal_conductivity = 1.,
        liquid_viscosity = 1.,
        gravity = (0., -1.),
        m_B = m_B,
        ddT_m_B = ddT_m_B,
        mesh = fenics.UnitSquareMesh(fenics.dolfin.mpi_comm_world(), grid_size, grid_size),
        initial_values_expression = ("0.", "0.", "0.", "0.5 - x[0]"),
        boundary_conditions = [{"subspace": 0,
                "value_expression": ("0.", "0."), "degree": 3,
                "location_expression": "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", "method": "topological"},
            {"subspace": 2,
                "value_expression": "0.5", "degree": 2, 
                "location_expression": "near(x[0],  0.)", "method": "topological"},
             {"subspace": 2,
                "value_expression": "-0.5", "degree": 2, 
                "location_expression": "near(x[0],  1.)", "method": "topological"}],
        end_time = end_time,
        time_step_size = time_step_size,
        stop_when_steady = stop_when_steady,
        steady_relative_tolerance=steady_relative_tolerance)
        
    return solution

solution = heat_driven_cavity()

verify_against_wang2010(solution)

end_time = 0.02

baseline_time_step_size = 1.e-3

baseline_grid_size = 40

baseline_solution = heat_driven_cavity(output_dir = "output/heat_driven_cavity_baseline",
        grid_size = baseline_grid_size,
        time_step_size = baseline_time_step_size,
        end_time = end_time)

def plot_temperature(solution):
    
    velocity, pressure, temperature = fenics.split(solution)

    fenics.plot(temperature)
    
    
plot_temperature(baseline_solution)

solutions = {"nt20_nx40": baseline_solution}

def compute_and_append_new_solution(nt, nx, solutions):

    solution = heat_driven_cavity(
        output_dir = "output/heat_driven_cavity/nt" + str(nt) + "/nx" + str(nx) + "/",
        grid_size = nx,
        time_step_size = end_time/nt,
        end_time = end_time)

    solutions["nt" + str(nt) + "_nx" + str(nx)] = solution
    
    return solutions

r = 2

baseline_time_step_count = int(round(end_time/baseline_time_step_size))

print(baseline_time_step_count)

nt = int(baseline_time_step_count/r)

nx = baseline_grid_size

solutions = compute_and_append_new_solution(nt, nx, solutions)

print(solutions.keys())

plot_temperature(solutions["nt10_nx40"])

solutions = compute_and_append_new_solution(int(nt/r), nx, solutions)

print(solutions.keys())

plot_temperature(solutions["nt5_nx40"])

errors = [fenics.errornorm(solutions["nt20_nx40"].split()[2], solutions["nt" + str(nt) + "_nx40"].split()[2], "L2") 
          for nt in [10, 5]]

print(errors)

import math

def compute_order(fine, coarse, r):
    
    return math.log(fine/coarse)/math.log(1./r)

order = compute_order(errors[0], errors[1], r)

print(order)

for i in [1, 2]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, nx, solutions)
    
print(solutions.keys())

errors = [fenics.errornorm(solutions["nt80_nx40"], solutions["nt" + str(nt) + "_nx40"], "L2") for nt in [40, 20, 10, 5]]

print(errors)

for i in range(len(errors) - 1):
    
    order = compute_order(errors[i], errors[i + 1], r)

    print(order)

for i in [3, 4]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, nx, solutions)
    
print(solutions.keys())

errors = [fenics.errornorm(solutions["nt320_nx40"].split()[2], solutions["nt" + str(nt) + "_nx40"].split()[2], "L2") 
          for nt in [160, 80, 40, 20, 10, 5]]

print(errors)

for i in range(len(errors) - 1):
    
    order = compute_order(errors[i], errors[i + 1], r)

    print(order)

for i in [5, 6]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, nx, solutions)

print(solutions.keys())

errors = [fenics.errornorm(solutions["nt1280_nx40"].split()[2], solutions["nt" + str(nt) + "_nx40"].split()[2], "L2") 
          for nt in [640, 320, 160, 80, 40, 20, 10, 5]]

print(errors)

for i in range(len(errors) - 1):
    
    order = compute_order(errors[i], errors[i + 1], r)

    print(order)



