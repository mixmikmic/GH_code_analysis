import msmexplorer as msme
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns 
sns.set_style("whitegrid")
sns.set_context("talk",1.3)
from msmbuilder.utils import load

plot_feat = load("./train_data/raw_features.pkl")
train_feat = load("./train_data/features.pkl")

df = load("./train_data/feature_descriptor.pkl")

df

scatter(np.vstack(plot_feat)[:,0],np.vstack(plot_feat)[:,1])
xlim([-pi,pi])
ylim([-pi,pi])

from sklearn.linear_model import LogisticRegression

import os

X=np.vstack(plot_feat)
train_X=np.vstack(train_feat)

train_Y=np.concatenate([np.zeros(len(plot_feat[0])),
            np.ones(len(plot_feat[0]))])
if not os.path.isfile("./lr_model_2.pkl"):
    train =True 
else:
    clf = load("./lr_model_2.pkl")
    train =False
if train:
    clf = LogisticRegression(penalty="l1",C=100)
    clf.fit(train_X, train_Y)

if train:
    from msmbuilder.utils import dump
    dump(clf,"./lr_model_2.pkl")

clf

lr_cv = b=clf.predict_proba(train_X)

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
       c=clf.predict_proba(train_X)[:,0],cmap='viridis')
xlim([-pi,pi])
ylim([-pi,pi])
cb=colorbar(p)
cb.set_label(r'$LR_{cv}$')
ylabel(r'$\psi$',size=26)
xlabel(r'$\phi$',size=26)

bar([0,1,2,3],clf.coef_[0],color=sns.color_palette("colorblind")[0])
xticks([0.4,1.5,2.5,3.5],[r'$\phi^{sin}$',r'$\phi^{cos}$',r'$\psi^{sin}$',r'$\psi^{cos}$'],size=20)
xlabel("Feature")
ylabel(r'$LR_{cv}$ Coefficients')


from sklearn.utils.validation import check_is_fitted

from tica_metadynamics.pyplumed import render_df
from tica_metadynamics.pyplumed import render_meta 
from jinja2 import Template

plumed_matheval_template = Template("MATHEVAL ARG={{arg}} FUNC={{func}} LABEL={{label}} PERIODIC={{periodic}} ")

plumed_combine_template = Template("COMBINE LABEL={{label}} ARG={{arg}} COEFFICIENTS={{coefficients}} "+                                    "PERIODIC={{periodic}} ")

def render_lr(clf=None, input_prefix="f0", output_prefix="l"):
    if clf is None or check_is_fitted(clf,attributes=["coef_","intercept_"]):
        raise ValueError("Need a fitted Sklearn Logistic Regression object")
    else:
        n_args = clf.coef_.shape[1]
        output = []
        arg_list=",".join(["%s_%d"%(input_prefix,i) for i in range(n_args)])
        coeff = ",".join([str(i) for i in clf.coef_[0]])
        w_norm = 1.0 
        
        output.append(plumed_combine_template.render(label="%s_0"%output_prefix,
                                      arg=arg_list,
                                      coefficients=coeff,
                                      periodic="NO")+"\n")
        
        func="1/(1+exp(-(x+%s)))"%(str(clf.intercept_[0]))
        
        output.append(plumed_matheval_template.render(label="%s_1"%output_prefix,
                                      arg="l_0",
                                      func=func,
                                      periodic="NO")+"\n")        
        
    return ''.join(output)
        
        

b=clf.predict_proba(train_X)
np.std(b[300:,0])

total_out=[]
total_out.extend("RESTART\n")
total_out.extend(render_df(df))
total_out.extend(render_lr(clf))
total_out.extend(render_meta.render_metad_code("l_1",biasfactor=8,sigma=0.05))
total_out.extend(render_meta.render_metad_bias_print("l_1,metad.bias"))

print("".join(total_out))

# Again, we can analyze the results after running simulations elsewhere

import mdtraj as md
from msmbuilder.featurizer import DihedralFeaturizer

test_traj = md.load("./lr_meta_traj_lowc/reweight//trajectory.dcd",top="./0.pdb")
feat = load("./train_data//featurizer.pkl")
bias = np.loadtxt("./lr_meta_traj_lowc/reweight//BIAS")

test_feat = feat.transform([test_traj])[0]
plot_test_feat = DihedralFeaturizer(sincos=False).transform([test_traj])[0]

clf = load("./lr_model_2.pkl")

test_feat

clf_out = clf.predict_proba(test_feat)

clf_out

plot(clf.predict_proba(test_feat)[:,1],bias[:,1])
xlabel("SKLearn Values")
ylabel("Plumed Values")

scatter(plot_test_feat[:,0],plot_test_feat[:,1],c=clf_out[:,1],cmap='viridis')
colorbar()

subplot(2,1,1)
plot(bias[:,1])
xticks([0,1000,2000,3000,4000],[0,10,20,30,40])
ylabel(r'$LR_{cv}$')

subplot(2,1,2)
plot(plot_test_feat[:,0])
xticks([0,1000,2000,3000,4000],[0,10,20,30,40])
xlabel("Simulation time (ns)")
ylim([-pi,pi])
ylabel(r'$\phi$')

# Or do some basic re-weighting using Tiwary's algorithm 

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











# These are from the scikit learn website and were used to make parts of figure 1 

from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
tmp_x = [xx[0],yy[0]]
b=np.dot(clf.coef_,tmp_x)+clf.intercept_/np.linalg.norm(clf.coef_)
print(b)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# this is our test set, it's just a straight line with some
# Gaussian noise
xmin, xmax = -5, 5
n_samples = 100
np.random.seed(0)
X = np.random.normal(size=n_samples)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=n_samples)

X = X[:, np.newaxis]
# run the classifier
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# and plot the result
plt.figure(1)
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)


def model(x):
    return 1 / (1 + np.exp(-x))
loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss,linestyle='dashed', linewidth=1,label="Logistic Regression")


plt.xlabel('Feature 1')
plt.ylabel('Probability Ouput')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(loc="lower right", fontsize='small')
plt.show()



