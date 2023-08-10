# We'll build up tables of t,x,v values by stepping forward in time.  
# First, initialize the tables to empty lists.

t_table = []
x_table = []
v_table = []

# The variables t,x,v will keep track of the current values of time,position,velocity 
# as we evolve the equations.  First we initialize them to starting values which are 
# determined by the initial conditions of the differential equation.

t = 0.0
x = 13.0
v = 0.0

# Now let's evolve the differential equations forward in time, to t=10 (say).
#
# We use a step size dt=0.01.  Choosing a proper step size can be tricky and sometimes
# requires trial-and-error experimentation.  If the step size is chosen too large, then
# the differential equations will be integrated inaccurately and the solution will look
# "jagged".  If the step size is chosen too small, then many steps will be needed and
# the calculation will take a long time (or crash the computer).
#
# In the following 'while' loop, the value of t will be increased by dt=0.01 in every
# iteration, and the loop exits when the value of t reaches 10.

while t <= 10.0:
    # Append the current values of t,x,v to the tables
    t_table.append(t)
    x_table.append(x)
    v_table.append(v)
    
    # Compute dx/dt and dv/dt from the differential equations
    dx_dt = v
    dv_dt = 5.*(10.-x)
    
    # Now update the values of t,x,v and 
    dt = 0.01             # step size
    t = t + dt            # t advances by "dt"
    x = x + dx_dt * dt    # x advances by its rate (dx/dt) times "dt"
    v = v + dv_dt * dt    # likewise for v
    

# After running the program above, the tables t_table, x_table, v_table will
# be populated with the solution to the differential equations.
# Let's print a few entries in the table, just to check that it worked!
print t_table[0]
print x_table[0]
print v_table[0]

print t_table[100]
print x_table[100]
print v_table[100]

# These commands only need to be included once per notebook.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Plot x versus t.
# Note that the amplitude of the oscillation appears to be increasing in the plot.
# This is incorrect since the true solution to the differential equation is a sinusoid.
# This is actually just an artifact of the numerical integration, and one could see this
# by rerunning with a smaller step size dt.

plt.plot(t_table, x_table)

# Plot v versus t.

plt.plot(t_table, v_table)



