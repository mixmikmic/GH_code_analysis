from pytriqs.operators import c, c_dag, n, Operator # n and Operator will be needed later
print c_dag('up',0)
print c('up',0)
print c_dag('down',0)
print c('down',0)

print n('up',0)

# Should give 0
print n('up',0) - c_dag('up',0)*c('up',0)

# Some calculation
print n('up',0) - 2 * c_dag('up',0)*c('up',0)

# Define the parameters
U = 4
mu = 3

# H is an empty operator
H = Operator()

# Add elements to define a Hamiltonian
H += U * n('up',0) * n('down',0)
H -= mu * (n('up',0) + n('down',0))
print H

