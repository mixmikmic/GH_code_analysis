from msmbuilder.example_datasets import AlanineDipeptide
from msmbuilder.featurizer import DihedralFeaturizer
get_ipython().run_line_magic('pylab', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid', palette='colorblind')
sns.set_context('talk',1.3)
import numpy as np
import sys,os,glob
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


sys.path.insert(0, "helper_func")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import torch as t
from torchvision.transforms import ToPILImage
from IPython.display import Image
to_img = ToPILImage()

import matplotlib.image as mpimg
plt.figure(figsize=(20,10))
images = []
for img_path in glob.glob('data/image_data/*.jpeg'):
    images.append(mpimg.imread(img_path))
    
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(images[i])
    #sns.despine()

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
data = ImageFolder(root='data', transform=ToTensor())

from torch.utils.data import DataLoader
loader = DataLoader(data)
for x, y in loader:
    print(x) # image
    print(y) # image label
    break

# We have a VERY basic 3 layerd network with an affine layer, a sigmoid non-linearity, and a final  Affine layer
# Image --> Affine --> Sigmoid --> Affine --> Score


# Hyper Parameters 
input_size = 3*64*64
hidden_size = 10
num_classes = 5

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        return out
    
net = Net(input_size, hidden_size, num_classes)

fake_top_list=[]
for i in range(64*64):
    fake_top_list.append({'serial':i,'name':"H1", "element":"H","resSeq":i,"resName":"H","chainID":"A"})
t2 = md.Topology.from_dataframe(pd.DataFrame(fake_top_list), None)

index = 0
loader = DataLoader(data)
for x, y in loader:
    xyz = np.array(x.view(-1,3*64*64).tolist()[0]).reshape(64*64,3)
    pred_y = net(Variable(x.view(-1,3*64*64)))
    if index == 0:
        all_pred = pred_y
        t1 = md.Trajectory(xyz=xyz, topology=t2)
    else:
        all_pred = torch.cat((all_pred,pred_y,))
        t1 += md.Trajectory(xyz=xyz, topology=t2)
    assert np.allclose(x.view(-1,3*64*64).tolist()[0],t1.xyz[-1].reshape(3*64*64))
    index += 1

t1.save_xtc("./images.xtc")

from torchvision.utils import make_grid

subplot(1,2,1)
title("Last PyTorch Image")
imshow(np.transpose(make_grid(x,padding=0).numpy(),(1,2,0)),interpolation='Nearest')
xlim([0,63])
ylim([0,63])
subplot(1,2,2)
title("Last Trajectory Frame")
imshow(np.transpose(t1.xyz[-1].reshape((3,64,64)),(1,2,0)),interpolation='Nearest')
xlim([0,63])
ylim([0,63])

nn_y = np.array(all_pred.data.tolist())

# Save network before something happens

torch.save(net,"imagenetwork.net")

output =  render_network(net)
with open("image_plumed.dat",'w') as f:
    f.writelines(output)

get_ipython().system('head image_plumed.dat')

get_ipython().system('tail image_plumed.dat')

# this creates the feature extractor 
def write_xyz_feature(t1):
    output = []
    for i in range(t1.n_atoms):
        output.append(plumed_position_templete.render(arg=i+1,label="l0%d"%(i)))
        output.append("\n")
    return ''.join(output)

# this creates a fully connected layer
def render_fc_layer(layer_indx, lp):
    output=[]
    for i in np.arange(lp.out_features):
        if layer_indx == 1:
            # we need the x,y,z(RGB) for each atom(pixel)
            arg = ','.join(["l0%d.%s"%(i,j) for i in range(4096) for j in ["x","y","z"]])
        else:
            # otherwise use the previous layer
            arg=','.join(["l%d%d"%(layer_indx-1,j) for j in range(lp.in_features)])
        
        weights = ','.join(map(str,lp.weight[i].data.tolist()))
        bias =','.join(map(str,lp.bias[i].data.tolist()))
        
        # combine without bias
        non_bias_label = "l%d%dnb"%(layer_indx, i)
        output.append(plumed_combine_template.render(arg = arg,
                                   coefficients = weights,
                                   label=non_bias_label,
                                   periodic="NO") +"\n")
        # now add the bias
        bias_label = "l%d%d"%(layer_indx, i)
        output.append(create_neural_bias(non_bias_label, bias, bias_label))
        output.append("\n")
    return ''.join(output)
    
    
# this cretes a sigmoid layer
def render_sigmoid_layer(layer_indx, lp, hidden_size=50):
    output=[]    
    for i in np.arange(hidden_size):
        arg="l%d%d"%(layer_indx-1, i)
        label = "l%d%d"%(layer_indx, i)
        output.append(create_sigmoid(arg, label))
        output.append("\n")
        
    return ''.join(output)

def render_network(net):
    output =[]
    # Start by evaluating the actual dihedrals + sin-cosine transform aka the input features 
    output.append(write_xyz_feature(t1))
    index = 0
    # Go over every layer of the netowrk
    for lp in net.children():
        index += 1
        if str(lp).startswith("Linear"):
            output.append(render_fc_layer(index, lp))
        elif str(lp).startswith("Sigmoid"):
            output.append(render_sigmoid_layer(index, lp,hidden_size=net.hidden_size))
        else:
            raise ValueError("Only Linear and Sigmoid Layers are supported for now")
    # Lastly, we want to print out the values from the last layer. This becomes our CV. 
    arg = ','.join(["l%d%d"%(index,j) for j in range(num_classes)])
    output.append(render_print_val(arg, file="ImageCV"))
    return ''.join(output)



#This command will generate the output for us
# plumed driver --mf_xtc images.xtc --plumed image_plumed.dat

# the 0th column in plumeds' CV file is always time so all outputs are off by 1
plumed_vals = np.loadtxt("ImageCV")

plt.figure(figsize=(16,13))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.title("Score for Label %d"%i)
    plt.scatter(nn_y[:,i],plumed_vals[:,i+1])
    assert np.allclose(nn_y[:,i],plumed_vals[:,i+1],1e-4,1e-4)
    plt.xlabel("PyTorch")
    plt.ylabel("Plumed")

from jinja2 import Template

plumed_torsion_template = Template("TORSION ATOMS={{atoms}} LABEL={{label}} ")

plumed_matheval_template = Template("MATHEVAL ARG={{arg}} FUNC={{func}} LABEL={{label}} PERIODIC={{periodic}} ")

plumed_combine_template = Template("COMBINE LABEL={{label}} ARG={{arg}} COEFFICIENTS={{coefficients}} "+                                    "PERIODIC={{periodic}} ")
plumed_print_template = Template("PRINT ARG={{arg}} STRIDE={{stride}} FILE={{file}} ")

#p: POSITION ATOM=3
plumed_position_templete = Template("POSITION ATOM={{arg}} LABEL={{label}}")

def create_torsion_label(inds, label):
    #t: TORSION ATOMS=inds
    return plumed_torsion_template.render(atoms=','.join(map(str, inds)), label=label) +"\n"


def create_feature(argument, func, feature_label):
    arg = argument
    x="x"
    if func in ["sin","cos"]:
        f = "%s(%s)"%(func,x)
        label = feature_label
    else:
        raise ValueError("Can't find function")

    return plumed_matheval_template.render(arg=arg, func=f,                                           label=label,periodic="NO")


def create_neural_bias(nb, bias, label):
    arg = ",".join([nb])
    f = "+".join(["x", bias])
    return plumed_matheval_template.render(arg=arg, func=f,                                           label=label,periodic="NO")
def create_sigmoid(arg, label):
    f = "1/(1+exp(-x))"
    return plumed_matheval_template.render(arg=arg, func=f,                                           label=label,periodic="NO")

def render_print_val(arg,stride=1,file="CV"):
    return plumed_print_template.render(arg=arg,
                                       stride=stride,
                                       file=file)

def get_feature_function(df, feature_index):
    possibles = globals().copy()
    possibles.update(locals())
    func = possibles.get("create_torsion_label")
    return func

def match_mean_free_function(df, feature_index):
    func = df.otherinfo[feature_index]
    return func











