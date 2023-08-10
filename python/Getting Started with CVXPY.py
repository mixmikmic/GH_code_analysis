import cvxpy as cvx
import numpy as np

# Create two scalar optimization variables.
x = cvx.Variable()
y = cvx.Variable()

# Create two constraints.
constraints = [3*x + 4*y == 26,
               2*x - 3*y == -11]

# Form objective.
obj = cvx.Minimize(cvx.norm(x+y))

# Form and solve problem.
prob = cvx.Problem(obj,constraints)
prob.solve()  # Returns the optimal value

# Display the solution

print("Status:", prob.status)
print("Objective Value:", prob.value)
print("Solution:", x.value, y.value)



