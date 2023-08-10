

# import skrf as rf
from matplotlib import pyplot as plt
from skrf.calibration import OnePort
from skrf.calibration.calibration import TRL
from skrf import data

#import your snp files from Electronics Desktop simulation
T = rf.Network('Lab3_thru.s2p')
R = rf.Network('Lab3_reflect.s1p')
L = rf.Network('Lab3_line.s2p')

#switch_terms = (rf.Network('trl_data/forward switch term.s1p'),
               # rf.Network('trl_data/reverse switch term.s1p'))

measured = [T,R,L]

