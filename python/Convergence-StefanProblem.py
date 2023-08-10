get_ipython().magic('matplotlib inline')

import fenics

import phaseflow

def stefan_problem(time_step_count = 100,
        grid_size = 1000):

    T_h = 1.
    
    T_c = -0.01
    
    end_time = 0.1
                       
    stefan_number = 0.125
    
    nt = time_step_count
    
    nx = grid_size
    
    solution, mesh = phaseflow.run(
        output_dir = "output/convergence_stefan_problem/nt" + str(nt) + "/nx" + str(nx) + "/",
        prandtl_number = 1.,
        stefan_number = stefan_number,
        gravity = [0.],
        mesh = fenics.UnitIntervalMesh(grid_size),
        initial_values_expression = (
            "0.",
            "0.",
            "(" + str(T_h) + " - " + str(T_c) + ")*(x[0] <= 0.001) + " + str(T_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3,
                 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_h, 'degree': 2, 
                 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': T_c, 'degree': 2, 
                 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = 0.,
        regularization_smoothing_factor = 0.01,
        end_time = end_time,
        time_step_size = end_time/float(time_step_count),
        stop_when_steady = False)
        
    velocity, pressure, temperature = solution.split()
        
    return temperature
    

baseline_time_step_count = 1

baseline_grid_size = 1000

baseline_solution = stefan_problem(
        time_step_count = baseline_time_step_count,
        grid_size = baseline_grid_size)

fenics.plot(baseline_solution)

solutions = {"nt1_nx1000": baseline_solution}

def compute_and_append_new_solution(nt, nx, solutions):

    solution = stefan_problem(
        time_step_count = nt,
        grid_size = nx)

    solutions["nt" + str(nt) + "_nx" + str(nx)] = solution
    
    fenics.plot(solution)
    
    print(solutions.keys())
    
    return solutions

r = 2

for i in [1, 2]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, baseline_grid_size, solutions)

errors = [fenics.errornorm(solutions["nt4_nx1000"], solutions["nt" + str(nt) + "_nx1000"], "L2") 
          for nt in [2, 1]]

print(errors)

import math

def compute_order(fine, coarse):
    
    return math.log(fine/coarse)/math.log(1./r)

order = compute_order(errors[0], errors[1])

print(order)

for i in [3, 4, 5]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, baseline_grid_size, solutions)

errors = [fenics.errornorm(solutions["nt32_nx1000"], solutions["nt" + str(nt) + "_nx1000"], "L2") 
          for nt in [16, 8, 4, 2, 1]]

print(errors)

for i in range(len(errors) - 1):

    order = compute_order(errors[i], errors[i + 1])

    print(order)

for i in [6, 7, 8]:
    
    solutions = compute_and_append_new_solution(r**i*baseline_time_step_count, baseline_grid_size, solutions)

errors = [fenics.errornorm(solutions["nt256_nx1000"], solutions["nt" + str(nt) + "_nx1000"], "L2") 
          for nt in [128, 64, 32, 16, 8, 4, 2, 1]]

print(errors)

for i in range(len(errors) - 1):

    order = compute_order(errors[i], errors[i + 1])

    print(order)

baseline_grid_size = 64

for j in [0, 1, 2]:
    
    solutions = compute_and_append_new_solution(nt = 16, nx = r**j*baseline_grid_size, solutions = solutions)

errors = [fenics.errornorm(solutions["nt16_nx256"], solutions["nt16" + "_nx" + str(nx)], "L2") 
          for nx in [128, 64]]

print(errors)

for i in range(len(errors) - 1):

    order = compute_order(errors[i], errors[i + 1])

    print(order)

for j in [3, 4, 5, 6]:
    
    solutions = compute_and_append_new_solution(nt = 16, nx = r**j*baseline_grid_size, solutions = solutions)

errors = [fenics.errornorm(solutions["nt16_nx4096"], solutions["nt16" + "_nx" + str(nx)], "L2") 
          for nx in [2048, 1024, 512, 256, 128, 64]]

print(errors)

for i in range(len(errors) - 1):

    order = compute_order(errors[i], errors[i + 1])

    print(order)

for j in [7, 8, 9]:
    
    solutions = compute_and_append_new_solution(nt = 16, nx = r**j*baseline_grid_size, solutions = solutions)

errors = [fenics.errornorm(solutions["nt16_nx16384"], solutions["nt16" + "_nx" + str(nx)], "L2") 
          for nx in [8192, 4096, 2048, 1024, 512, 256, 128, 64]]

print(errors)

for i in range(len(errors) - 1):

    order = compute_order(errors[i], errors[i + 1])

    print(order)



