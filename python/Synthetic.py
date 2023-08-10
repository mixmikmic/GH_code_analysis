import numpy as np
import matplotlib.pyplot as plt

class Adam:

    def __init__(self, x0, amsgrad=False, alpha=0.001, beta1=0.9, beta2=0.999,
                 F=None, epsilon=1e-8, decay=False):
        self.x = x0

        self.amsgrad = amsgrad
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.F = F
        self.epsilon = epsilon
        self.decay = decay

        self.m = 0
        self.v = 0
        self.t = 0
        self.vmax = 0

    def step(self, grad):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        if self.amsgrad:
            self.vmax = max(self.vmax, self.v)
            m = self.m
            v = self.vmax
        else:
            m = self.m / (1 - self.beta1**self.t)
            v = self.v / (1 - self.beta2**self.t)

        lr = self.alpha / np.sqrt(self.t) if self.decay else self.alpha
        x_updated = self.x - lr * m / (np.sqrt(v) + self.epsilon)

        self.x = x_updated if self.F is None else np.clip(x_updated, self.F[0], self.F[1])
        return self.x

def experiment(modulo=101, iterations = 7000000, F=(-1,1), max_grad=1010, min_grad=-10,
               online_graph=True, stochastic_graph=True,
               online_params=(0.01, False), stochastic_params=(0.5, True)):

    adam_o = Adam(0, F=F, alpha=online_params[0], decay=online_params[1])
    amsg_o = Adam(0, amsgrad=True, F=F, alpha=online_params[0], decay=online_params[1])
    adam_s = Adam(1, F=F, alpha=stochastic_params[0], decay=stochastic_params[1])
    amsg_s = Adam(1, amsgrad=True, F=F, alpha=stochastic_params[0], decay=stochastic_params[1])

    ts = range(1, iterations + 1)
    xs_adam_o, xs_amsg_o = [], []
    xs_adam_s, xs_amsg_s = [], []

    for t in ts:    
        # Gradients
        grad_o = (max_grad if t % modulo == 1 else min_grad)
        grad_s = (max_grad if np.random.random() < 0.01 else min_grad)

        # Update the xs
        x_adam_o = adam_o.step(grad_o)
        x_amsg_o = amsg_o.step(grad_o)
        x_adam_s = adam_s.step(grad_s)
        x_amsg_s = amsg_s.step(grad_s)

        xs_adam_o.append(x_adam_o)
        xs_amsg_o.append(x_amsg_o)
        xs_adam_s.append(x_adam_s)
        xs_amsg_s.append(x_amsg_s)

    if online_graph:
        plt.title('Value of x at each iteration (online setting)')
        plt.xlabel('iterations')
        plt.ylabel('xt')
        plt.plot(ts, xs_adam_o, label='ADAM')
        plt.plot(ts, xs_amsg_o, label='AMSgrad')
        plt.legend()
        plt.show()

    if stochastic_graph:
        plt.title('Value of x at each iteration (stochastic setting)')
        plt.xlabel('iterations')
        plt.ylabel('xt')
        plt.plot(ts, xs_adam_s, label='ADAM')
        plt.plot(ts, xs_amsg_s, label='AMSgrad')
        plt.legend()
        plt.show()

experiment(modulo=101)

experiment(modulo=11, stochastic_graph=False)

experiment(modulo=301, stochastic_graph=False)

experiment(modulo=100, stochastic_graph=False)

experiment(modulo=102, stochastic_graph=False)

experiment(modulo=101, online_params=(0.5, True), stochastic_params=(0.25, True))

experiment(modulo=101, online_params=(0.75, True), stochastic_params=(0.75, True))

