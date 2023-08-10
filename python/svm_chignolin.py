# some basic python imports 
import msmexplorer as msme
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns 
sns.set_style("whitegrid")
sns.color_palette("colorblind")
sns.set_context("poster",1.3)
import mdtraj as md
from msmbuilder.utils import load,dump

loc = "./data/"

all_atom_trj = md.load("%s/0.pdb"%loc)

all_atom_f = load("./data/all_atom_featurizer.pkl")
all_atom_df = load("./data/all_atom_feature_descriptor.pkl")

all_atom_df.head(10)

nrm = load("./data/nrm.pkl")

from sklearn.svm import SVC
import os

train=False 
if not os.path.isfile("./data//svm_model.pkl"):
    clf = SVC(kernel="linear")
    train =True 
else:
    clf = load("./data/svm_model.pkl")
    train =False

# Unfortunately, we can't provide the actual features but the trainging for these models is the same as alanine 
# except we now use a second transform 

# here basin_!.hdf5 is the trajectory from the unfolded state, and basin_2.hdf5 is the trajecotory from the folded state

# t1 = md.load("./basin_1.hdf5")
# t2 = md.load("./basin_2.hdf5")

# features = all_atom_f.fit_transform([t1,t2])
# nrm_features = nrm.fit_transform(np.concatenate(features))

# train_X=np.vstack(nrm_features)

# train_Y=np.concatenate([np.zeros(1000),
#             np.ones(1000)])
# if train:
#     clf.fit(train_X, train_Y)
# else:
#     pass



all_atom_df.iloc[np.argsort(np.abs(clf.coef_))[0]]

clr_plt = sns.color_palette("colorblind")
plot(clf.coef_.T,marker='o',c=clr_plt[2])
vlines(14,-.15,0.15,linestyles='dashed')
vlines(50,-.15,0.15,linestyles='dashed')
ylim([-.15,0.15])
xlabel("Feature Index")
ylabel(r'SVM coefficient')

# in these train_x is from above
# b=clf.decision_function(train_X)/np.linalg.norm(clf.coef_)
# np.std(b[1000:])

plot(b[:1000],label="Unfolded state",c=clr_plt[0])
plot(b[1000:],label="Folded state",c=clr_plt[1])
legend()
xlabel("Training simulation time (ns)")
ylabel(r'$SVM_{cv}$')
xticks([0,200,400,600,800,1000],[0,400,800,1200,1600,2000])
ylim([-12,6])

from tica_metadynamics.pyplumed import render_df
from tica_metadynamics.pyplumed import render_meta 
from jinja2 import Template
from sklearn.utils.validation import check_is_fitted

plumed_matheval_template = Template("MATHEVAL ARG={{arg}} FUNC={{func}} LABEL={{label}} PERIODIC={{periodic}} ")

plumed_combine_template = Template("COMBINE LABEL={{label}} ARG={{arg}} COEFFICIENTS={{coefficients}} "+                                    "PERIODIC={{periodic}} ")
def render_svm(clf=None, input_prefix="f0", output_prefix="l"):
    if clf is None or check_is_fitted(clf,attributes=["coef_","intercept_"]):
        raise ValueError("Need a fitted Sklearn SVM object")
    else:
        n_args = clf.coef_.shape[1]
        output = []
        arg_list=",".join(["%s_%d"%(input_prefix,i) for i in range(n_args)])
        coeff = ",".join([str(i) for i in clf.coef_[0]])
        w_norm = np.linalg.norm(clf.coef_)
        
        output.append(plumed_combine_template.render(label="%s_0"%output_prefix,
                                      arg=arg_list,
                                      coefficients=coeff,
                                      periodic="NO")+"\n")
        
        func="(x+%s)/%s"%(str(clf.intercept_[0]),str(w_norm))
        
        output.append(plumed_matheval_template.render(label="%s_1"%output_prefix,
                                      arg="l_0",
                                      func=func,
                                      periodic="NO")+"\n")        
        
    return ''.join(output)
        
        

total_out=[]
total_out.extend("RESTART\n")
total_out.extend(render_df(all_atom_df,nrm=nrm))
total_out.extend(render_svm(clf))
total_out.extend(render_meta.render_metad_code("l_1",biasfactor=6,sigma=0.25))
total_out.extend(render_meta.render_metad_bias_print("l_1,metad.bias"))

print("".join(total_out))

# We can now analyze the multiple walker simulations in differnt ways

# for example, we can concatenate all trajectories and load the bias for each frame in  that single long traj



loc = "./data/reweight/"
test_traj = md.load("%s/all_traj.xtc"%loc,top="%s/top.pdb"%loc)
bias = np.loadtxt("%s/BIAS"%loc)
#bias = np.loadtxt("./reweight2/reweight//BIAS")
#bias = np.loadtxt("./reweight//BIAS")

test_X = nrm.transform(all_atom_f.transform([test_traj])[0])
sklearn_out = clf.decision_function(test_X)/np.linalg.norm(clf.coef_)


plot(sklearn_out, label="ALL WALKERS")
legend()

plot(sklearn_out,bias[:,1])
xlabel("SKLearn Values")
ylabel("Plumed Values")

clr_plt = sns.color_palette("colorblind")
ax,data=msme.plot_free_energy(bias,obs=[1],n_samples=50000,pi=np.exp(bias[:,-1]/2.83),
                      cmap='viridis',gridsize=400,return_data=True,shade=False,color=clr_plt[5])
ax.set_ylim([0,3])
xlabel(r'$SVM_{cv}$'+"\nUnfolded to Folded")
ylabel("Free Energy (kcal/mol)")

# Or simply sum up the hills

fes = np.loadtxt("./%s/fes.dat"%loc)

plot(fes[:,0],(fes[:,1]-fes[:,1].min())/4.18,c=clr_plt[5])
ylim([0,4])
xlabel(r'$SVM_{cv}$'+"\nUnfolded to Folded")
ylabel("Free Energy (kcal/mol)")

folded_pdb = md.load_pdb("https://files.rcsb.org/view/2RVD.pdb")

folded_pdb_decision_func = clf.decision_function(np.concatenate([nrm.transform(i) for i in all_atom_f.transform(folded_pdb)]))                      /np.linalg.norm(clf.coef_)

folded_pdb_decision_func

ca_traj = test_traj.atom_slice([i.index for i in test_traj.top.atoms if i.name=='CA'])
ca_folded_pdb= folded_pdb.atom_slice([i.index for i in folded_pdb.top.atoms if i.name=='CA'])

rmsd_data = md.rmsd(ca_traj,ca_folded_pdb)
plot(rmsd_data,label="ALL WALKERS")
legend()
xlabel("Time (ns)")
xticks([0,10000,20000,30000,40000,50000,60000],np.array([0,10000,20000,30000,40000,50000,60000])/2)
ylabel("RMSD to folded (nm)")

rmsd_dict={}
for walker_index in range(25):
    test_traj = md.load("%s/walker_%d.xtc"%(loc,walker_index),top="%s/top.pdb"%loc)
    ca_traj = test_traj.atom_slice([i.index for i in test_traj.top.atoms if i.name=='CA'])
    ca_folded_pdb= folded_pdb.atom_slice([i.index for i in folded_pdb.top.atoms if i.name=='CA'])
    rmsd_data = md.rmsd(ca_traj,ca_folded_pdb)
    rmsd_dict[walker_index] = rmsd_data

# quick hack to find interesting trajectories 

np.argsort([np.median(rmsd_dict[i]) for i in range(25)])

for i in [24,10,14,15,8]:
    plot(rmsd_dict[i])

walker_index = 24
test_traj = md.load("%s/walker_%d.xtc"%(loc,walker_index),top="%s/top.pdb"%loc)
test_X = nrm.transform(all_atom_f.transform([test_traj])[0])
sklearn_out = clf.decision_function(test_X)/np.linalg.norm(clf.coef_)

subplot(2,1,1)
plot(sklearn_out[:600],c=clr_plt[1],label="Walker 25 (of 25)")
ylabel(r'$SVM_{cv}$')
legend()
xticks([0,200,400,600],[])
ylim([-12,9])
subplot(2,1,2)
plot(rmsd_dict[24][:600],c=clr_plt[5],label="Walker 25 (of 25)")
legend()
xticks([0,200,400,600],[0,10,20,30])
xlabel("Metdynamics simulation time (ns)")
ylabel("RMSD(nm) to folded")
# vlines(585,0.1,0.8,linestyles='dotted')





























