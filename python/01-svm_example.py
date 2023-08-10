# this notebook is a walkthrough on how to build collective variables using Supervised machine learning algorithms

# some imports 
import msmexplorer as msme
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns 
sns.set_style("whitegrid")
sns.set_context("poster",1.3)
from msmbuilder.utils import load

plot_feat = load("./train_data/raw_features.pkl")
train_feat = load("./train_data/features.pkl")

df = load("./train_data/feature_descriptor.pkl")

# The pandas data frame tells us what the features are that we are using
df

from sklearn.svm import LinearSVC,SVC
import os

if not os.path.isfile("./svm_model_2.pkl"):
    clf = svc = LinearSVC(penalty="l1",C=1,dual=False)
    train =True 
else:
    clf = load("./svm_model_2.pkl")
    train =False

print(clf)

X=np.vstack(plot_feat)
train_X=np.vstack(train_feat)

y=np.concatenate([np.zeros(len(plot_feat[0])),
            np.ones(len(plot_feat[0]))])

if train:
    clf.fit(train_X,y)

train_X.sum(axis=1)[300:].std()

nx =ny=50
lim_x = lim_y = np.linspace(-pi,pi,nx)
xv, yv = np.meshgrid(lim_x, lim_y, sparse=False, indexing='ij')
res = []
for i in range(nx):
    for j in range(ny):
        X_val = np.array([np.sin(xv[i,j]),np.cos(xv[i,j]), np.sin(yv[i,j]), np.cos(yv[i,j])]).reshape(1,-1)
        res.extend(clf.predict(X_val))
#contourf(lim_x,lim_y,np.array(res).reshape(10,10),cmap='coolwarm')

contourf(lim_x,lim_y,np.array(res).reshape(nx,ny).T,cmap="coolwarm",alpha=0.3)
p=scatter(np.vstack(plot_feat)[:,0],np.vstack(plot_feat)[:,1],
       c=clf.decision_function(train_X)/np.linalg.norm(clf.coef_),cmap='viridis')
xlim([-pi,pi])
ylim([-pi,pi])
cb=colorbar(p)
cb.set_label(r'$SVM_{cv}$')
ylabel(r'$\psi$',size=26)
xlabel(r'$\phi$',size=26)

train

if train:
    from msmbuilder.utils import dump
    dump(clf,"./svm_model_2.pkl")

b=clf.decision_function(train_X)/np.linalg.norm(clf.coef_)
np.std(b[300:])

bar([0,1,2,3],clf.coef_[0],color=sns.color_palette("colorblind")[0])
xticks([0.4,1.5,2.5,3.5],[r'$\phi^{sin}$',r'$\phi^{cos}$',r'$\psi^{sin}$',r'$\psi^{cos}$'],size=20)
xlabel("Feature")
ylabel(r'$SVM_{cv}$ Coefficients')


from sklearn.utils.validation import check_is_fitted



# These imports are designed to interface between msmbuilder and Plumed
from tica_metadynamics.pyplumed import render_df
from tica_metadynamics.pyplumed import render_meta 
from jinja2 import Template

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
total_out.extend(render_df(df))
total_out.extend(render_svm(clf))
total_out.extend(render_meta.render_metad_code("l_1",biasfactor=8,sigma=0.1))
total_out.extend(render_meta.render_metad_bias_print("l_1,metad.bias"))

print("".join(total_out))

# Run the metadynamics on a GPU

import mdtraj as md
from msmbuilder.featurizer import DihedralFeaturizer

test_traj = md.load("./svm_meta_traj_lowc/reweight//trajectory.dcd",top="./0.pdb")
feat = load("./train_data//featurizer.pkl")
bias = np.loadtxt("./svm_meta_traj_lowc/reweight//BIAS")

test_traj

test_feat = feat.transform([test_traj])[0]
plot_test_feat = DihedralFeaturizer(sincos=False).transform([test_traj])[0]

clf_out =clf.decision_function(test_feat)/np.linalg.norm(clf.coef_)

clf

scatter(clf_out,bias[:,1])
xlabel("SKLearn Values")
ylabel("Plumed Values")

clf

subplot(2,1,1)
plot(bias[:,1])
xticks([0,1000,2000,3000,4000],[0,10,20,30,40])
ylabel(r'$SVM_{cv}$')

subplot(2,1,2)
plot(plot_test_feat[:,0])
xticks([0,1000,2000,3000,4000],[0,10,20,30,40])
xlabel("Simulation time (ns)")
ylim([-pi,pi])
ylabel(r'$\phi$')

scatter(plot_test_feat[:,0],plot_test_feat[:,1],c=clf_out,cmap='viridis')
colorbar()

ax,data=msme.plot_free_energy(plot_test_feat,obs=[0,1],n_samples=100000,pi=np.exp(bias[:,-1]/2.49),
                      cmap='viridis',gridsize=100,vmin=-1,vmax=7,n_levels=8,return_data=True)

offset = data[2].min(0)
contour(data[0],data[1],data[2],levels=np.linspace(-1,5,7))
p=contourf(data[0],data[1],data[2],levels=np.linspace(-1,5,7),cmap='viridis')
cb=colorbar(p)
cb.set_label("Free Energy (kcal/mol)")
xlabel(r'$\phi$',size=26)
ylabel(r'$\psi$',size=26)
xlim([-pi,pi])
ylim([-pi,pi])









