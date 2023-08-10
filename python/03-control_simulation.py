import msmexplorer as msme
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns 
sns.set_style("whitegrid")
sns.set_context("poster",1.3)
from msmbuilder.utils import load

import mdtraj as md
from msmbuilder.featurizer import DihedralFeaturizer

title_list = ["Control Simulation("+r'$\psi_{CV}$)',r'$SVM_{CV}$',r'$DNN_{CV}$']

for i,f in enumerate(["control_meta_traj_1","svm_meta_traj_lowc","dnn_meta_traj"]):
    subplot(3,1,i+1)
    test_traj = md.load("%s//reweight//trajectory.dcd"%f,top="./0.pdb")
    plot_test_feat = DihedralFeaturizer(sincos=False).transform([test_traj])[0]
    plot(plot_test_feat[:1200,0],label=title_list[i],c=sns.color_palette("colorblind")[i])

    if i==2:
        xticks([0,300,600,900,1200],[0,3,6,9,12])
        xlabel("Simulation time (ns)")
    else:
        xticks([0,500,1000,1200],[])
    ylim([-pi,pi])
    ylabel(r'$\phi$')
    legend()

title_list = ["Control Simulation("+r'$\phi_{CV}$)',r'$SVM_{CV}$',r'$DNN_{CV}$']

for i,f in enumerate(["control_meta_traj_2","svm_meta_traj_lowc","dnn_meta_traj"]):
    subplot(3,1,i+1)
    test_traj = md.load("%s//reweight//trajectory.dcd"%f,top="./0.pdb")
    plot_test_feat = DihedralFeaturizer(sincos=False).transform([test_traj])[0]
    plot(plot_test_feat[:1200,0],label=title_list[i],c=sns.color_palette("colorblind")[i])

    if i==2:
        xticks([0,300,600,900,1200,1500],[0,3,6,9,12,15])
        xlabel("Simulation time (ns)")
    else:
        xticks([0,500,1000,1200,1500],[])
    ylim([-pi,pi])
    ylabel(r'$\phi$')
    legend()

scatter(plot_test_feat[:,0],plot_test_feat[:,1])







