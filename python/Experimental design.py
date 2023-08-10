import numpy as np
from collections import Counter

new_zealand = {'Ncont': 6021, 'Xcont': 302, 'Nexp': 5979, 'Xexp': 374}
other = {'Ncont': 50000, 'Xcont': 2500, 'Nexp': 50000, 'Xexp': 2500}

global_ = dict(Counter(new_zealand) + Counter(other))

def p(cont_or_exp, data):
    return data['X{}'.format(cont_or_exp)] / data['N{}'.format(cont_or_exp)]

def p_pool(data):
    return (data['Xcont'] + data['Xexp']) / (data['Ncont'] + data['Nexp'])

def se_pool(data):
    return np.sqrt(p_pool(data) * (1-p_pool(data)) * ((1 / data['Ncont']) + (1 / data['Nexp'])))

print('The pooled global p is {}'.format(p_pool(global_)))

print('The pooled global SE is {}'.format(se_pool(global_)))

print('The estimated difference globally is {}'.format(p('exp', global_) - p('cont', global_)))

print('The margin of error is {}'.format(1.96 * se_pool(global_)))

print('The pooled new zealand p is {}'.format(p_pool(new_zealand)))
print('The pooled global SE is {}'.format(se_pool(new_zealand)))
print('The estimated difference globally is {}'.format(p('exp', new_zealand) - p('cont', new_zealand)))
print('The margin of error is {}'.format(1.96 * se_pool(new_zealand)))



