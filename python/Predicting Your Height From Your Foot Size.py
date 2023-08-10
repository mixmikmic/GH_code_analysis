get_ipython().run_line_magic('pylab', 'inline')
from mpl_toolkits.mplot3d import axes3d

## Load the data
def read_data():
    data_file = open("india_foot_height.dat","r")
    raw_data = data_file.readlines()
    data_file.close()
    return [line.strip().split("  ") for line in raw_data]

## Store the data for further functions
fh_data = read_data()
print("Number of Data Points:{}".format(len(fh_data)))
## Plot the data
plot([x[0] for x in fh_data],[x[1] for x in fh_data],'ro',color='orange',marker='p')

def error(b,m):
    totalError = 0
    for point in fh_data:
        x = float(point[0])
        y = float(point[1])
        totalError += (y - m*x - b)**2
    return totalError/float(len(fh_data))

def step_gradient(current_b,current_m):
    de_dm = 0
    de_db = 0
    N = float(len(fh_data))
    for point in fh_data:
        x = float(point[0])
        t = float(point[1])
        de_dm += (-2/N)*(t - current_m * x + current_b)*(x)
        de_db += (-2/N)*(t - current_m * x + current_b)
    new_b = current_b - de_db*learning_rate
    new_m = current_m - de_dm*learning_rate
    
    return [new_b,new_m]

learning_rate = 0.0001
iterations = 1000
# Define Some Variables For Plotting
errors = []
b_values = []
m_values = []

def gradient_descent(i_b,i_m):
    b = i_b
    m = i_m
    for i in range(iterations):
        b,m = step_gradient(b,m)
        errors.append(error(b,m))
        b_values.append(b)
        m_values.append(m)
    return [b,m]

def run():
    initial_b = 0
    initial_m = 0
    print("Initial m:{} & initial b:{}".format(initial_m,initial_b))
    plot([x[0] for x in fh_data],[x[1] for x in fh_data],'ro',color='orange')
    
    print("Initial Error:{}".format(error(initial_b,initial_m)))
    [b,m] = gradient_descent(initial_b,initial_m)
    print("After {} iterations, final b: {},final m:{}".format(iterations,b,m))
    print("Final Error: {}".format(error(b,m)))
    plot([x for x in range(22,29)],[(m*x + b) for x in range(22,29)],color='blue')
    return [b,m]

[b,m] = run()

# Lets Plot Error Vs Iteration
len(errors)
plot(errors[:50])

fig = figure()
ax = fig.gca(projection='3d')
X = (m_values)
Y = (b_values)
Z = (errors)
X, Y = meshgrid(X, Y)
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
ax.view_init(10,80)
draw()

def find_my_height(shoesize):
    """
    Input: Shoesize in cm
    Output: Height in cm
    """
    return (m*shoesize + b)

find_my_height(27) 

