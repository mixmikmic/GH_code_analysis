KALDI_ROOT='~/apps/kaldi'

import os
from shutil import copyfile
import fileinput
import stat

KALDI_ROOT=os.path.expanduser(KALDI_ROOT)
KALDI_ROOT=os.path.abspath(KALDI_ROOT)

if not os.getcwd().endswith('/work'):
    if not os.path.exists('work'):
        os.mkdir('work')
    os.chdir('work')

if not os.path.exists('steps'):
    os.symlink(KALDI_ROOT+'/egs/wsj/s5/steps','steps')

if not os.path.exists('path.sh'):
    copyfile(KALDI_ROOT+'/egs/wsj/s5/path.sh','path.sh')
    for line in fileinput.input('path.sh',inplace=True):
        if line.startswith('export KALDI_ROOT='):
            print 'export KALDI_ROOT='+KALDI_ROOT
        else:
            print line[:-1]
    os.chmod('path.sh',0755)

import sys

sys.path.append('steps/nnet3')

import nnet3_train_lib as ntl

path = ntl.RunKaldiCommand('source ./path.sh ; printenv | grep ^PATH=')[0]

print 'Setting '+path

os.environ['PATH']=path.split('=')[1]

def problem(x):
        return x[0]*0.3+x[1]*0.1+0.2

import numpy as np

np.random.seed(1234)

input_dim=2
data_num=100

inputs=np.random.random((data_num,input_dim))
outputs=np.array([problem(x) for x in inputs])
if outputs.ndim==1:
        outputs=outputs.reshape(outputs.shape[0],1)

def write_simple_egs(filename,inputs,outputs):
    input_dim=inputs.shape[1]
    output_dim=outputs.shape[1]
    with open(filename,'w') as f:
        for i,l in enumerate(inputs):
            
            f.write('data-{} '.format(i))
            f.write('<Nnet3Eg> ')
            f.write('<NumIo> {} '.format(input_dim))

            f.write('<NnetIo> input ')
            f.write('<I1V> 1 <I1> 0 0 0  ')
            f.write('[\n  ')
            for d in l:
                f.write('{} '.format(d))
            f.write(']\n')
            f.write('</NnetIo> ')

            f.write('<NnetIo> output ')
            f.write('<I1V> 1 <I1> 0 0 0  ')
            f.write('[\n  ')
            for d in outputs[i]:
                f.write('{} '.format(d))
            f.write(']\n')
            f.write('</NnetIo> ')

            f.write('</Nnet3Eg> ')
            
write_simple_egs('nnet.egs',inputs,outputs)

get_ipython().system('head -n 10 nnet.egs')

get_ipython().run_cell_magic('writefile', 'nnet.config', '# First the components\ncomponent name=wts type=AffineComponent input-dim=2 output-dim=1 learning-rate=0.6\n# Next the nodes\ninput-node name=input dim=2\ncomponent-node name=wts_node component=wts input=input\noutput-node name=output input=wts_node objective=quadratic')

print ntl.RunKaldiCommand('nnet3-init {} {}'.format('nnet.config','nnet.init'))[1]

print ntl.RunKaldiCommand('nnet3-info {}'.format('nnet.init'))[0]

from IPython.display import SVG, Image, display
from subprocess import check_call

ntl.RunKaldiCommand('nnet3-info {} | python steps/nnet3/dot/nnet3_to_dot.py {}'.format('nnet.init','nnet.dot'))
#my installation of dot doesn't support PNG, so I have to resort to SVG
check_call(['dot','-Tsvg','nnet.dot','-o','nnet.svg'])
#SVG can't be scaled in notebook, but I can use imagemagick to convert to PNG
check_call(['convert','nnet.svg','nnet.png'])
display(Image('nnet.png'))

print ntl.RunKaldiCommand('nnet3-copy --binary=false {} {}'.format('nnet.init','-'))[0]

print ntl.RunKaldiCommand('nnet3-train {} ark,t:{} {}'.format('nnet.init','nnet.egs','nnet.out'))[1]

print ntl.RunKaldiCommand('nnet3-copy --binary=false {} {}'.format('nnet.out','-'))[0]
for i in range(3):
    ntl.RunKaldiCommand('nnet3-train {} ark,t:{} {}'.format('nnet.out','nnet.egs','nnet.out'))
    print ntl.RunKaldiCommand('nnet3-copy --binary=false {} {}'.format('nnet.out','-'))[0]

test_num=10

test=np.random.random((test_num,input_dim))

print test

for x in test:
    print problem(x)

with open('test.mat','w') as f:
    f.write('test [')
    for row in test:
        f.write('\n  ')
        f.write(' '.join([`num` for num in row]))
    f.write('  ]\n')
get_ipython().magic('cat test.mat')

print ntl.RunKaldiCommand('nnet3-compute {} ark,t:{} ark,t:{}'.format('nnet.out','test.mat','-'))[0]



