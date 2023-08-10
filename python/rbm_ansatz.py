from __future__ import division
try:
    import torch
except:
    get_ipython().system('pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl ')
    import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def vmc_sample(kernel, initial_config, num_bath, num_sample):
    '''
    obtain a set of samples.

    Args:
        kernel (object): defines how to sample, requiring the following methods:
            * propose_config, propose a new configuration.
            * prob, get the probability of specific distribution.
        initial_config (1darray): initial configuration.
        num_bath (int): number of updates to thermalize.
        num_sample (int): number of samples.

    Return:
        list: a list of spin configurations.
    '''
    print_step = np.Inf  # steps between two print of accept rate, Inf to disable showing this information.

    config = initial_config
    prob = kernel.prob(config)

    n_accepted = 0
    sample_list = []
    for i in range(num_bath + num_sample):
        # generate new config and calculate probability ratio
        config_proposed = kernel.propose_config(config)
        prob_proposed = kernel.prob(config_proposed)

        # accept/reject a move by metropolis algorithm (world's most famous single line algorithm)
        if np.random.random() < prob_proposed / prob:
            config = config_proposed
            prob = prob_proposed
            n_accepted += 1

        # print statistics
        if i % print_step == print_step - 1:
            print('%-10s Accept rate: %.3f' %
                  (i + 1, n_accepted * 1. / print_step))
            n_accepted = 0

        # add a sample
        if i >= num_bath:
            sample_list.append(config_proposed)
    return sample_list


def vmc_measure(local_measure, sample_list, measure_step, num_bin=50):
    '''
    perform measurements on samples

    Args:
        local_measure (func): local measurements function, input configuration, return local energy and local gradient.
        sample_list (list): a list of spin configurations.
        num_bin (int): number of bins in binning statistics.
        meaure_step: number of samples skiped between two measurements + 1.

    Returns:
        tuple: expectation valued of energy, gradient, energy*gradient and error of energy.
    '''
    # measurements
    energy_loc_list, grad_loc_list = [], []
    for i, config in enumerate(sample_list):
        if i % measure_step == 0:
            # back-propagation is used to get gradients.
            energy_loc, grad_loc = local_measure(config)
            energy_loc_list.append(energy_loc)
            grad_loc_list.append(grad_loc)

    # binning statistics for energy
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    # get expectation values
    energy_loc_list = torch.from_numpy(energy_loc_list)
    if grad_loc_list[0][0].is_cuda: energy_loc_list = energy_loc_list.cuda()
    grad_mean = []
    energy_grad = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = torch.stack(grad_loc, 0)
        grad_mean.append(grad_loc.mean(0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.dim() - 1)] * grad_loc).mean(0))
    return energy.item(), grad_mean, energy_grad, energy_precision



def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    num_sample = len(var_list)
    if num_sample % num_bin != 0:
        raise
    size_bin = num_sample // num_bin

    # mean, variance
    mean = np.mean(var_list, axis=0)
    variance = np.var(var_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var(
        [np.mean(var_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin *         np.abs(np.mean(variance_binned) / np.mean(variance))
    stderr = np.sqrt(variance_binned / num_bin)
    print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
          (mean, stderr, t_auto))
    return mean, stderr

class VMCKernel(object):
    '''
    variational monte carlo kernel.

    Attributes:
        energy_loc (func): local energy <x|H|\psi>/<x|\psi>.
        ansatz (Module): torch neural network.
    '''
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc

    def prob(self, config):
        '''
        probability of configuration.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number: probability |<config|psi>|^2.
        '''
        return abs(self.ansatz.psi(torch.from_numpy(config)).data[0])**2

    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number, list: local energy and local gradients for variables.
        '''
        psi_loc = self.ansatz.psi(torch.from_numpy(config))

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        psi_loc.backward()
        grad_loc = [p.grad.data/psi_loc.data[0] for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.ansatz.psi(torch.from_numpy(x)).data, psi_loc.data)[0]
        return eloc, grad_loc

    @staticmethod
    def propose_config(old_config, prob_flip=0.05):
        '''
        flip two positions as suggested spin flips.

        Args:
            old_config (1darray): spin configuration, which is a [-1,1] string.
            prob_flip (float): the probability to flip all spins, to make VMC more statble in Heisenberg model.

        Returns:
            1darray: new spin configuration.
        '''
        # take ~ 5% probability to flip all spin, can making VMC sample better in Heisenberg model
        if np.random.random() < prob_flip:
            return -old_config

        num_spin = len(old_config)
        upmask = old_config == 1
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = -1
        config[iflip1] = 1
        return config

class RBM(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible).double() * 1e-1)
        self.v_bias = nn.Parameter(torch.zeros(num_visible).double())
        self.h_bias = nn.Parameter(torch.randn(num_hidden).double() * 1e-1)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
    def psi(self, v):
        '''
        probability for visible nodes, visible/hidden nodes here take value in {-1, 1}.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v = Variable(v.double())
        if self.W.is_cuda: v = v.cuda()
        res = (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()
        return res

def heisenberg_loc(config, psi_func, psi_loc, J=1.):
    '''
    local energy for 1D Periodic Heisenberg chain.

    Args:
        config (1darray): bit string as spin configuration.
        psi_func (func): wave function.
        psi_loc (number): wave function projected on configuration <config|psi>.
        J (float): coupling strengh. Here, J = Jxy = Jz.

    Returns:
        number: local energy.
    '''
    # get weights and flips after applying hamiltonian \sum_i w_i|x_i> = H|x>
    nsite = len(config)
    wl, flips = [], []
    # J*SzSz terms.
    nn_par = np.roll(config, -1) * config
    wl.append(J / 4. * (nn_par).sum(axis=-1).item())
    flips.append(np.array([], dtype='int64'))

    # J*SxSx and J*SySy terms.
    mask = nn_par != 1
    i = np.where(mask)[0]
    j = (i + 1) % nsite
    wl += [-J / 2.] * len(i)
    flips.extend(zip(i, j))

    # calculate local energy <\psi|H|x>/<\psi|x>
    acc = 0.
    for wi, flip in zip(wl, flips):
        config_i = config.copy()
        config_i[list(flip)] *= -1
        eng_i = wi * (psi_func(config_i) / psi_loc)
        acc += eng_i
    return acc

def train(model, learning_rate, use_cuda):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    initial_config = np.array([-1, 1] * (model.ansatz.num_visible // 2))
    if use_cuda:
        model.ansatz.cuda()

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.        
        sample_list = vmc_sample(model, initial_config, num_bath=200, num_sample=8000)
        energy, grad, energy_grad, precision = vmc_measure(model.local_measure, sample_list, num_spin)

        # update variables using steepest gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(model.ansatz.parameters(), g_list):
            delta = learning_rate * g
            var.data -= delta
        yield energy, precision

get_ipython().run_line_magic('matplotlib', 'inline')
import time
import matplotlib.pyplot as plt

# set random number seed
use_cuda = False
seed = 10086
torch.manual_seed(seed)
if use_cuda: torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
max_iter = 100
num_spin = 8
num_hidden = 16
E_exact = -3.65109341

# visualize the loss history
energy_list, precision_list = [], []
def _update_curve(energy, precision):
    energy_list.append(energy)
    precision_list.append(precision)
    if len(energy_list)%10 == 0:
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list, capsize=3)
        # dashed line for exact energy
        plt.axhline(E_exact, ls='--')
        plt.show()

rbm = RBM(num_spin, num_hidden)
model = VMCKernel(heisenberg_loc, ansatz=rbm)

t0 = time.time()
for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1, use_cuda = use_cuda)):
    t1 = time.time()
    print('Step %d, dE/|E| = %.4f, elapse = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
    _update_curve(energy, precision)
    t0 = time.time()

    # stop condition
    if i >= max_iter:
        break

def J1J2_loc(config, psi_func, psi_loc, J1, J1z, J2, J2z, periodic):
    '''
    local energy for 1D J1, J2 chain.

    Args:
        config (1darray): bit string as spin configuration.
        psi_func (func): wave function.
        psi_loc (number): wave function projected on configuration <config|psi>.
        J1, J1z, J2, J2z (float): coupling strenghs for nearest neightbor, nearest neighbor z-direction, 2nd nearest neighbor and 2nd nearest neighbor in z-direction.
        periodic (bool): boundary condition.

    Returns:
        number: local energy.
    '''
    nsite = len(config)
    wl, flips = [], []
    Js = [J1, J2]
    Jzs = [J1z, J2z]
    for INB, (J,Jz) in enumerate(zip(Js, Jzs)):
        # J1(SzSz) terms.
        nn_par = np.roll(config, -INB-1) * config
        if not periodic:
            nn_par = nn_par[:-INB-1]
        wl.append(Jz / 4. * (nn_par).sum(axis=-1))
        flips.append(np.array([], dtype='int64'))

        # J1(SxSx) and J1(SySy) terms.
        mask = nn_par != 1
        i = np.where(mask)[0]
        j = (i + INB+1) % nsite

        wl += [J / 2.] * len(i)
        flips.extend(zip(i, j))
        
    # calculate local energy <\psi|H|x>/<\psi|x>
    acc = 0
    for wi, flip in zip(wl, flips):
        config_i = config.copy()
        config_i[list(flip)] *= -1
        eng_i = wi * psi_func(config_i) / psi_loc
        acc += eng_i
    return acc





