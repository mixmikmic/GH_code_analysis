from pybrain.optimization import GA

# sum of squares
def objective(x):
    return sum(x**2)

x0 = [100.,500.]

learning = GA(objective,x0,minimize=True)
learning.learn()

def objective(x):
    return abs(sum(x))
x0 = [500.]

GA(objective, [20.], minimize=True).learn()

from scipy.optimize import rosen
from pybrain.optimization import GA
GA(rosen,[10.,10.],minimize=True).learn()



