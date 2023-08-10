# this notebook is a walkthrough on how to build collective variables using Supervised machine learning algorithms

# some imports 
import msmexplorer as msme
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns 
sns.set_style("whitegrid")
sns.set_context("poster",1.3)
from msmbuilder.utils import load

import mdtraj as md 

from msmbuilder.featurizer import DihedralFeaturizer

t1 = md.load("../SML_CV/alanine_example/svm_meta_traj_mc/train_traj.xtc",
            top="../SML_CV/alanine_example/svm_meta_traj_mc/train_traj_top.pdb")

plot_feat = DihedralFeaturizer(sincos=False).transform([t1])[0]

# beta basin
basin_1_inds = np.where(np.logical_and(plot_feat[:,0]<0,plot_feat[:,1]>2))[0][:100]
# alpha_r basin
basin_2_inds = np.where(np.logical_and(plot_feat[:,0]<0,                                       np.logical_and(plot_feat[:,1]<1,plot_feat[:,1]>-1)))[0][:100]
#alpha_l
basin_3_inds = np.where(np.logical_and(plot_feat[:,0]>0,                                       np.logical_and(plot_feat[:,1]<2,plot_feat[:,1]>-1)))[0][:100]

# scatter(plot_feat[:,0],plot_feat[:,1],alpha=0.3)
for v,i in enumerate([basin_1_inds,basin_2_inds,basin_3_inds]):
    scatter(plot_feat[i,0],plot_feat[i,1],c=sns.color_palette()[v])

train_traj = t1[basin_1_inds] +t1[basin_2_inds] +t1[basin_3_inds]

train_feat = DihedralFeaturizer().transform([train_traj])[0] 
plot_feat=DihedralFeaturizer(sincos=False).transform([train_traj])[0]

import pandas as pd

df = pd.DataFrame(DihedralFeaturizer().describe_features(train_traj))

# The pandas data frame tells us what the features are that we are using
df

from sklearn.svm import LinearSVC,SVC
import os

if not os.path.isfile("./svm_model_3.pkl"):
    clf = svc = LinearSVC(penalty="l1",C=1,dual=False)
    train =True 
else:
    clf = load("./svm_model_3.pkl")
    train =False

print(clf)

plot_feat.shape

X=np.vstack(plot_feat)
train_X=np.vstack(train_feat)

y=np.concatenate([np.zeros(100),np.ones(100),np.ones(100)+1])

if train:
    clf.fit(train_X,y)

nx =ny=50
lim_x = lim_y = np.linspace(-pi,pi,nx)
xv, yv = np.meshgrid(lim_x, lim_y, sparse=False, indexing='ij')
res = []
for i in range(nx):
    for j in range(ny):
        X_val = np.array([np.sin(xv[i,j]),np.cos(xv[i,j]), np.sin(yv[i,j]), np.cos(yv[i,j])]).reshape(1,-1)
        res.extend(clf.predict(X_val))
#contourf(lim_x,lim_y,np.array(res).reshape(10,10),cmap='coolwarm')

np.argmax(clf.decision_function(train_X),axis=1)



contourf(lim_x,lim_y,np.array(res).reshape(nx,ny).T,cmap="coolwarm",alpha=0.3)
p=scatter(np.vstack(plot_feat)[:100,0],np.vstack(plot_feat)[:100,1],
       c=sns.color_palette("colorblind")[0])
p=scatter(np.vstack(plot_feat)[100:200,0],np.vstack(plot_feat)[200:300,1],
       c=sns.color_palette("colorblind")[1])
p=scatter(np.vstack(plot_feat)[200:,0],np.vstack(plot_feat)[200:,1],
       c=sns.color_palette("colorblind")[2])
xlim([-pi,pi])
ylim([-pi,pi])
# cb=colorbar(p)
#cb.set_label(r'$SVM_{cv}$')
ylabel(r'$\psi$',size=26)
xlabel(r'$\phi$',size=26)

get_ipython().run_line_magic('pinfo', 'clf.decision_function')

if train:
    from msmbuilder.utils import dump
    dump(clf,"./svm_model_3.pkl")

b=clf.decision_function(train_X)

label_list=[r'$\beta$ vs rest',r'$\alpha_R$ vs rest',r'$\alpha_L$ vs rest']

plot([0,1,2,3],clf.coef_[0],color=sns.color_palette("colorblind")[0],marker='o',
    label=label_list[0])

plot([0,1,2,3],clf.coef_[1],color=sns.color_palette("colorblind")[1],marker='o',label=label_list[1])
plot([0,1,2,3],clf.coef_[2],color=sns.color_palette("colorblind")[2],marker='o',label=label_list[2])
xticks([0,1,2,3],[r'$\phi^{sin}$',r'$\phi^{cos}$',r'$\psi^{sin}$',r'$\psi^{cos}$'],size=20)
xlabel("Feature")
ylabel(r'$SVM_{cv}$ Coefficients')
legend(loc='best')

from sklearn.utils.validation import check_is_fitted



# These imports are designed to interface between msmbuilder and Plumed
from tica_metadynamics.pyplumed import render_df
from tica_metadynamics.pyplumed import render_meta 
from jinja2 import Template

plumed_matheval_template = Template("MATHEVAL ARG={{arg}} FUNC={{func}} LABEL={{label}} PERIODIC={{periodic}} ")

plumed_combine_template = Template("COMBINE LABEL={{label}} ARG={{arg}} COEFFICIENTS={{coefficients}} "+                                    "PERIODIC={{periodic}} ")

clf.classes_

def render_svm(clf=None, input_prefix="f0", output_prefix="l"):
    if clf is None or check_is_fitted(clf,attributes=["coef_","intercept_"]):
        raise ValueError("Need a fitted Sklearn SVM object")
    else:
        n_args = clf.coef_.shape[1]
        output = []
        arg_list=",".join(["%s_%d"%(input_prefix,i) for i in range(n_args)])
        if clf.classes_.shape[0] ==2 :
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
        else:
            for index,l_out in enumerate(clf.classes_):
                coeff = ",".join([str(i) for i in clf.coef_[index]])
                w_norm = np.linalg.norm(clf.coef_[index])

                output.append(plumed_combine_template.render(label="%s_%d0"%(output_prefix,l_out),
                                              arg=arg_list,
                                              coefficients=coeff,
                                              periodic="NO")+"\n")

                func="(x+%s)/%s"%(str(clf.intercept_[index]),str(w_norm))

                output.append(plumed_matheval_template.render(label="%s_%d1"%(output_prefix,l_out),
                                              arg="l_%d0"%l_out,
                                              func=func,
                                              periodic="NO")+"\n")
        
    return ''.join(output)
        
        

total_out=[]
total_out.extend("RESTART\n")
total_out.extend(render_df(df))
total_out.extend(render_svm(clf))
#total_out.extend(render_meta.render_metad_code("l_1",biasfactor=8,sigma=0.1))
#total_out.extend(render_meta.render_metad_bias_print("l_1,metad.bias"))

print("".join(total_out))

# Run the metadynamics on a GPU

import mdtraj as md
from msmbuilder.featurizer import DihedralFeaturizer

test_traj = md.load("./svm_meta_traj_mc/reweight//trajectory.dcd",top="./0.pdb")
feat = load("./train_data//featurizer.pkl")
bias = np.loadtxt("./svm_meta_traj_mc/reweight//BIAS")

test_traj

test_feat = feat.transform([test_traj])[0]
plot_test_feat = DihedralFeaturizer(sincos=False).transform([test_traj])[0]

clf_out =clf.decision_function(test_feat)/np.linalg.norm(clf.coef_,axis=1)

clf

figure(figsize=(12,4))
for i in range(3):
    subplot(1,3,i+1)
    scatter(clf_out[:,i],bias[:,i+1])
    xlabel("SKLearn Values")
    ylabel("Plumed Values")

subplot(4,1,1)
plot(bias[:1200,1],c=sns.color_palette("colorblind")[0])
xticks([0,400,800,1200],[])
ylabel(label_list[0])

subplot(4,1,2)
plot(bias[:1200,2],c=sns.color_palette("colorblind")[1])
xticks([0,400,800,1200],[])
ylabel(label_list[1])


subplot(4,1,3)
plot(bias[:1200,3],c=sns.color_palette("colorblind")[2])
xticks([0,400,800,1200],[])
ylabel(label_list[2])

subplot(4,1,4)
plot(plot_test_feat[:1200,0])
xticks([0,400,800,1200],[0,4,8,12])
xlabel("Simulation time (ns)")
ylim([-pi,pi])
ylabel(r'$\phi$')

scatter(plot_test_feat[:,0],plot_test_feat[:,1],c=clf_out[:,0],cmap='viridis')
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

get_ipython().run_line_magic('matplotlib', 'nbagg')
import matplotlib.animation as animation

get_ipython().run_line_magic('matplotlib', 'inline')

label_list=[r'$\beta$',r'$\alpha_R$',r'$\alpha_L$']

sns.set_context("paper")

all_tic_0 = np.arange(1200)

f0 = figure()#, dpi = 100)
ax01 = plt.subplot2grid((3, 4), (0, 0),colspan=2)
p01, = ax01.plot(bias[:1200,1],c=sns.color_palette("colorblind")[0],alpha=0.1)
p01, = ax01.plot(all_tic_0,bias[:1200,1],c=sns.color_palette("colorblind")[0])
ax01.set_xticks([0,400,800,1200],)
ax01.set_xticklabels([])
ax01.set_ylabel(label_list[0])


ax02 = plt.subplot2grid((3, 4), (1, 0),colspan=2)
p02, = ax02.plot(bias[:1200,2],c=sns.color_palette("colorblind")[1],alpha=0.1)
p02, = ax02.plot(all_tic_0,bias[:1200,2],c=sns.color_palette("colorblind")[1])
ax02.set_xticks([0,400,800,1200],)
ax02.set_xticklabels([])
ax02.set_ylabel(label_list[1])


ax03 = plt.subplot2grid((3, 4), (2, 0),colspan=2)
p03, = ax03.plot(bias[:1200,2],c=sns.color_palette("colorblind")[2],alpha=0.1)
p03, = ax03.plot(all_tic_0,bias[:1200,2],c=sns.color_palette("colorblind")[2])
ax03.set_xticks([0,400,800,1200],)
ax03.set_xticklabels([0,4,8,12])
ax03.set_ylabel(label_list[2])
ax03.set_xlabel("Simulation Time (ns)")



ax04 = plt.subplot2grid((3, 4), (1, 2),rowspan=2,colspan=2)
ax04.contourf(lim_x,lim_y,np.array(res).reshape(nx,ny).T,cmap="coolwarm",alpha=0.3)
p04,=ax04.plot(plot_test_feat[0,0],plot_test_feat[0,1])
ax04.set_xlabel(r'$\phi$')
ax04.set_ylabel(r'$\psi$')
ax04.set_ylim([-pi,pi])
ax04.set_xlim([-pi,pi])




def updateData(i):
    print(i)

    frm_pt = max(0, i-10)
    p01.set_data(all_tic_0[frm_pt:i], bias[frm_pt:i,1])
    p01.set_color(sns.color_palette("colorblind")[0])
    
    p02.set_data(all_tic_0[frm_pt:i], bias[frm_pt:i,2])#,c=sns.color_palette("colorblind")[1])
    p02.set_color(sns.color_palette("colorblind")[1])

    p03.set_data(all_tic_0[frm_pt:i], bias[frm_pt:i,3])#c=sns.color_palette("colorblind")[2])
    p03.set_color(sns.color_palette("colorblind")[2])

    p04.set_data(plot_test_feat[frm_pt:i,0], plot_test_feat[frm_pt:i,1])
    p04.set_color(sns.color_palette("colorblind")[4])
    p04.set_marker("o")

    return p01,p02,p03,p04

print(len(all_tic_0))
# f0.tight_layout()
f0.show()
simulation = animation.FuncAnimation(f0, updateData, np.arange(500),
    blit=False, interval=5,repeat=False)

simulation.save(filename='ala.mp4',fps=25,dpi=100,bitrate=5000)



















f0 = figure(num = 0, figsize = (9, 10))#, dpi = 100)
ax01 = f0.add_subplot(1,1,1)
ax01.grid('on')
ax01.set_ylim(-10,5)
ax01.set_xlim(0,600)
ax.set_xticks([0,100,200,300,400,500,600])
ax01.set_xlabel("Simulation time (ns)")
ax01.set_ylabel(r'$SVM_{cv}$')


all_tic_0 = np.arange(len(sklearn_out))
all_tic_1 = sklearn_out
tic0=np.zeros(0)
tic1=np.zeros(0)
t=np.zeros(0)

ax01.plot(sklearn_out,c=sns.color_palette("deep")[2],alpha=0.3,label="Walker 25 (of 25)")
ax01.legend(loc='lower right')
p014, = ax01.plot(tic0,tic1, marker='o',c=sns.color_palette("deep")[2])

ax01.set_xticklabels([0,5,10,15,20,25,30])

def updateData(i):
    print(i)

    frm_pt = max(0, i-100)
    p014.set_data(all_tic_0[frm_pt:i], all_tic_1[frm_pt:i])
    return p014,

print(len(all_tic_0))

f0.tight_layout()
simulation = animation.FuncAnimation(f0, updateData, np.arange(len(100)),
    blit=False, interval=5,repeat=False)
simulation.save(filename='t2_t.mp4',fps=25,dpi=100,bitrate=5000)
































