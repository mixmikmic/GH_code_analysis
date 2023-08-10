get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import simtk.openmm as mm
from torsionfit.database import qmdatabase as ScanSet
import torsionfit.model as Model
from torsionfit.backends import sqlite_plus
from pymc import MCMC
from parmed.charmm import CharmmParameterSet
from torsionfit import plots
from matplotlib.backends.backend_pdf import PdfPages
import torsionfit.parameters as par
from collections import OrderedDict
import numpy as np
from pymbar.timeseries import detectEquilibration
import tqdm

param_to_opt=[('CG331', 'CG321', 'CG321', 'CG331')]
param = CharmmParameterSet('../../../../data/charmm_ff/top_all36_cgenff.rtf', 
                           '../../../../data/charmm_ff/par_all36_cgenff.prm')
structure = '../../../structure/butane.psf'
scan = '../../../torsion_scans/MP2_torsion_scan/'
# Print initial guess

# Create a butane scan with torsions on (to compare for difference potential)
butane_scan = ScanSet.parse_psi4_out(scan, structure)
optimized = butane_scan.remove_nonoptimized()
optimized.compute_energy(param)

# Turn off torsion
param.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][1].phi_k=0
param.dihedral_types[('CG331', 'CG321', 'CG321', 'CG331')][0].phi_k=0

# Create butane scan with torsions off
optimized_0 = butane_scan.remove_nonoptimized()
optimized_0.compute_energy(param)

# Load samplers
dbs = OrderedDict()
for i in range(0, 40):
    dbs['db_{}'.format(i)] = sqlite_plus.load('tau_nuisance_0.01_{}/tau_nuisance_0.01_{}.sqlite'.format(i, i))

# Difference potential
# Difference potential
# look at difference potential
new_param = CharmmParameterSet('../../../../data/charmm_ff/top_all36_prot.rtf',
                              '../../../../data/charmm_ff/par_all36_cgenff.prm')
new_struct = butane_scan.remove_nonoptimized()
# parameterize with db_0
par.add_missing(param_list=param_to_opt, param=new_param, sample_n5=True)
with PdfPages('difference_potential_tau_10000.pdf') as pdf:
    # Update parameters
    for db in dbs:
        plt.figure()
        par.update_param_from_sample(param_list=param_to_opt, param=new_param, db=dbs[db], n_5=True, rj=True)

        # Check
        print(new_param.dihedral_types[param_to_opt[0]])    
        new_struct.compute_energy(new_param)
        plt.plot(optimized_0.angles, optimized_0.delta_energy-min(optimized_0.delta_energy), label='Torsion off')
        plt.plot(optimized.angles, optimized.delta_energy-min(optimized.delta_energy), label='Torsion on')
        plt.plot(new_struct.angles, new_struct.delta_energy - min(new_struct.delta_energy), label='Fit')
        plt.legend()
        plt.xlabel('Dihedral Angle (Degrees)')
        plt.ylabel('Relative Energy (KJ/mol)')
        plt.title('Butane Difference potential for {}. tau=nuisance'.format(db))
        pdf.savefig()

# Difference potential
# look at difference potential
new_param = CharmmParameterSet('../../../../data/charmm_ff/top_all36_prot.rtf',
                              '../../../../data/charmm_ff/par_all36_cgenff.prm')
new_struct = butane_scan.remove_nonoptimized()
# parameterize with db_0
par.add_missing(param_list=param_to_opt, param=new_param, sample_n5=True)
par.set_phase_0(param_list=param_to_opt, param=new_param)
with PdfPages('difference_potential_all_tau_nuisance.pdf') as pdf:
    # Update parameters
    plt.plot(optimized_0.angles, optimized_0.delta_energy-min(optimized_0.delta_energy), label='Torsion off', color='red')
    plt.plot(optimized.angles, optimized.delta_energy-min(optimized.delta_energy), label='Torsion on', color='green')
    for db in dbs:
        par.update_param_from_sample(param_list=param_to_opt, param=new_param, db=dbs[db], n_5=True, rj=True)

        # Check
        new_struct.compute_energy(new_param)
        plt.plot(new_struct.angles, new_struct.delta_energy - min(new_struct.delta_energy), alpha=0.2, color='blue',
                linewidth=5)
        plt.xlabel('Dihedral Angle (Degrees)')
        plt.ylabel('Relative Energy (KJ/mol)')
        plt.title('Butane Difference potential, tau=nuisance')
    plt.legend()
    pdf.savefig()

with PdfPages('deviance_tau_nuisance.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.figure()
        plt.plot((dbs[db].deviance[100:]), label='deviance')
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Energy (kJ/mol)')
        plt.title("Deviance {}, tau=nuisance".format(db))
        pdf.savefig()

with PdfPages('deviance_all_tau_nuisance.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.plot(dbs[db].deviance[1000:], alpha=0.3, label=i)
        #lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.xlabel('time')
        plt.ylabel('Deviance')
    plt.title("Deviance all tau=nuisance")
    pdf.savefig()

with PdfPages('multiplicity_bitstrings_tau_nuisance.pdf') as pdf: 
    bins = np.linspace(0, 65, 64)
    for db in dbs:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Multiplicity bitstring {}, tau=nuisance'.format(db))
        plt.plot(dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[:])
        plt.ylabel('Interger')
        plt.xlabel('Samples')
        plt.subplot(2, 1, 2)
        plt.hist(dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[1000:], bins=bins)
        plt.xticks(np.linspace(0, 64, 65), [0,'','','', '', 5, '', '', '', '', 10,'', '', '', '', 15,'', '', '', '',
                                        20,'', '', '', '', 25,'', '', '', '', 30, '', '', '', '',35,'', '', '', '',
                                        40,'', '', '', '', 45,'', '', '', '', 50,'', '', '', '', 55,'', '', '', '', 
                                        60, '', '', '', 64,])
        plt.xlabel('Interger')
        plt.ylabel('Frequency of int')
        plt.tight_layout()
        plt.tight_layout()
        pdf.savefig()

alpha=0.3
with PdfPages('multiplicity_all_tau_nuisance.pdf') as pdf: 
    bins = np.linspace(0, 65, 64)
    for db in dbs:
        plt.subplot(2, 1, 1)
        plt.title('Multiplicity bitstring tau=nuisance')
        plt.plot(dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[:], alpha=alpha, color='blue')
        plt.ylabel('Interger')
        plt.xlabel('Samples')
        plt.subplot(2, 1, 2)
        plt.hist(dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[:], bins=bins, alpha=alpha, color='blue')
        plt.xticks(np.linspace(0, 64, 65), [0,'','','', '', 5, '', '', '', '', 10,'', '', '', '', 15,'', '', '', '',
                                        20,'', '', '', '', 25,'', '', '', '', 30, '', '', '', '',35,'', '', '', '',
                                        40,'', '', '', '', 45,'', '', '', '', 50,'', '', '', '', 55,'', '', '', '', 
                                        60, '', '', '', 64,])
        plt.xlabel('Interger')
        plt.ylabel('Frequency of int')
        plt.tight_layout()
        plt.tight_layout()
    pdf.savefig()

from scipy import stats
mode = np.zeros((len(dbs)))
var = np.zeros((len(dbs)))
for i, db in enumerate(dbs):
    mode[i] = stats.mode(dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[3000:])[0]
    var[i] = dbs[db].CG331_CG321_CG321_CG331_multiplicity_bitstring[3000:].var()
plt.hist(mode, bins=int(mode.max()))
plt.title('Multiplicity mode, tau=nuisance')
plt.savefig('multiplicity_mode_tau_nuisance.pdf')

model = Model.TorsionFitModel(param, optimized_0, rj=True,
                                            param_to_opt=param_to_opt, sample_n5=True)
samples = 10000
burn = 0
multiplicities = (1,2,3,4,5,6)
mult_bitstring = []
for i in model.pymc_parameters.keys():
    if i.split('_')[-1] == 'bitstring':
        mult_bitstring.append(i)

samples = 10000
burn = 0
multiplicities = (1,2,3,4,5,6)
mult_bitstring = []
for i in model.pymc_parameters.keys():
    if i.split('_')[-1] == 'bitstring':
        mult_bitstring.append(i)
        
with PdfPages('marg_mult_equil_tau_nuisance.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.figure()

        histogram = np.zeros((len(mult_bitstring), samples, 6))

        for m, torsion in enumerate(mult_bitstring):
            for i, j in enumerate(dbs[db].trace('%s' % torsion)[5000:]):
                for k, l in enumerate(multiplicities):
                    if 2**(l-1) & int(j):
                        histogram[m][i][k] = 1

        plt.matshow(histogram.sum(1), cmap='Blues',  extent=[0, 5, 0, 20]), plt.colorbar()
        plt.yticks([])
        plt.xlabel('multiplicity term')
        plt.ylabel('torsion {}'.format(db))
        pdf.savefig()

alpha=0.5               
with PdfPages('hist_traces_all_tau_nuisance.pdf') as pdf:
    for n, db in enumerate(dbs):
        plt.figure()
        histogram = np.zeros((len(mult_bitstring), samples, 6))
        for m, torsion in enumerate(mult_bitstring):
            for i, j in enumerate(dbs[db].trace('%s' % torsion)[:]):
                for k, l in enumerate(multiplicities):
                    if 2**(l-1) & int(j):
                        histogram[m][i][k] = 1
        for t in param_to_opt:
            t_name = t[0] + '_' + t[1] + '_' + t[2] + '_' + t[3]
            for m in multiplicities:
                name = t_name + '_' + str(m) + '_K'
                plt.subplot(3, 1, 1)
                plt.plot(histogram[0, :, m-1], label=str(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                trace = dbs[db].trace(name)[:]
                index_0 = np.where(histogram[0, :, m-1] == 0)
                index_1 = np.where(histogram[0, :, m-1] == 1)
                np.place(trace, histogram[0, :, m-1] == 0, np.ones(len(index_0[0]))*np.nan)
                plt.subplot(3, 1, 2)
                plt.plot(trace, label='K_{}'.format(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.subplot(3, 1, 3)
                plt.hist(trace[index_1], label='K_{}'.format(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
        pdf.savefig(bbox_extra_artists=(lgd,), bbox_inches='tight')

alpha=0.5               
with PdfPages('hist_traces_all_tau_nuisance.pdf') as pdf:
    for n, db in enumerate(dbs):
        plt.figure()
        histogram = np.zeros((len(mult_bitstring), samples, 6))
        for m, torsion in enumerate(mult_bitstring):
            for i, j in enumerate(dbs[db].trace('%s' % torsion)[:]):
                for k, l in enumerate(multiplicities):
                    if 2**(l-1) & int(j):
                        histogram[m][i][k] = 1
        for t in param_to_opt:
            t_name = t[0] + '_' + t[1] + '_' + t[2] + '_' + t[3]
            for m in multiplicities:
                name = t_name + '_' + str(m) + '_K'
                plt.subplot(3, 1, 1)
                plt.plot(histogram[0, 1000:, m-1], label=str(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                trace = dbs[db].trace(name)[5000:]
                index_0 = np.where(histogram[0, 5000:, m-1] == 0)
                index_1 = np.where(histogram[0, 5000:, m-1] == 1)
                np.place(trace, histogram[0, 5000:, m-1] == 0, np.ones(len(index_0[0]))*np.nan)
                plt.subplot(3, 1, 2)
                plt.plot(trace, label='K_{}'.format(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.subplot(3, 1, 3)
                plt.hist(trace[index_1], label='K_{}'.format(m), alpha=alpha)
                lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
        pdf.savefig(bbox_extra_artists=(lgd,), bbox_inches='tight')

with PdfPages('tau.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.figure()
        plt.plot((dbs[db].precision_k[:]), label='precision')
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('precision')
        plt.title("Precision vs time db_{}".format(db))
        pdf.savefig()

with PdfPages('sigma.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.figure()
        plt.plot((dbs[db].sigma_k[:]), label='sigma')
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('sigma')
        plt.title("sigma vs time db_{}".format(db))
        pdf.savefig()

with PdfPages('sigma_all_tau_nuisance.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.plot(dbs[db].sigma_k[1000:], alpha=0.3, label=i)
        #lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.xlabel('time')
        plt.ylabel('sigma')
    plt.title("sigma for k")
    pdf.savefig()

with PdfPages('tau_all_tau_nuisance.pdf') as pdf:
    for i, db in enumerate(dbs):
        plt.plot(dbs[db].sigma[1000:], alpha=0.3, label=i)
        #lgd = plt.legend(prop={'size': 8}, loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.axvline(t_dev[i,0], color='red')
        #plt.text(t_dev[i, 0]+10, 8, r'$t_{equil}$', color='red', fontsize=20)
        plt.xlabel('time')
        plt.ylabel('sigma')
    plt.title("sigma for k")
    pdf.savefig()



