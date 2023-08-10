import numpy
import scipy.signal
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

class TankSystem:
    def __init__(self, A, alpha, K, V, Fi):
        """ This special function gets called when an object of this class is created"""
        self.A = A
        self.alpha = alpha
        self.K = K
        self.Fi = Fi
        self.change_state(V)
    
    def f(self, x):
        return self.alpha**(x - 1)
    
    def change_input(self, x):
        self.Fo = self.K*self.f(x)*numpy.sqrt(self.h)
        
    def change_state(self, V):
        self.state = self.V = V
        self.output = self.h = self.V/self.A

    def derivative(self, x):
        self.change_input(x)
        dVdt = self.Fi - self.Fo
        return dVdt

class PIController:
    def __init__(self, Kc, tau_i, bias):
        self.G = scipy.signal.lti([Kc*tau_i, Kc], [tau_i, 0])
        self.change_state(numpy.zeros((self.G.A.shape[0], 1)))
        self.bias = self.output = bias
        self.y = self.bias
        
    def change_input(self, u):
        self.y = self.G.C.dot(self.x) + self.G.D.dot(u) + self.bias
        self.output = self.y[0, 0]  # because y is a matrix, and we want a scalar output
    
    def change_state(self, x):
        self.x = self.state = x
    
    def derivative(self, e):
        return self.G.A.dot(self.x) + self.G.B.dot(e)

ts = numpy.linspace(0, 100, 1000)
dt = ts[1]

sp = 1.3

def control_simulation(system, controller):
    outputs = []
    for t in ts:
        system.change_input(controller.output)

        e = sp - system.output

        controller.change_input(e)

        system.change_state(system.state + system.derivative(controller.output)*dt)
        controller.change_state(controller.state + controller.derivative(e)*dt)

        outputs.append(system.output)
    return outputs

system = TankSystem(A=2, alpha=20, K=2, V=2, Fi=1)
controller = PIController(Kc=-1, tau_i=5, bias=0.7)

outputs = control_simulation(system, controller)

plt.plot(ts, outputs)

outputs = control_simulation(system=TankSystem(A=2, alpha=10, K=2, V=2, Fi=1), 
                             controller=PIController(Kc=-2, tau_i=5, bias=0.5))
plt.plot(ts, outputs);

class LtiSystem:
    def __init__(self, numerator, denominator):
        self.G = scipy.signal.lti(numerator, denominator)
        self.change_state(numpy.zeros((self.G.A.shape[0], 1)))
        self.y = self.output = 0
        
    def change_input(self, u):
        self.y = self.G.C.dot(self.x) + self.G.D.dot(u)
        self.output = self.y[0, 0]
    
    def change_state(self, x):
        self.x = self.state = x
    
    def derivative(self, e):
        return self.G.A.dot(self.x) + self.G.B.dot(e)

outputs = control_simulation(system=LtiSystem(1, [1, 1]), 
                             controller=PIController(Kc=1, tau_i=10, bias=0))
plt.plot(ts, outputs)



