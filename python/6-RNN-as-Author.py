import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph

import blocks.algorithms
from blocks.algorithms import GradientDescent
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model

from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.main_loop import MainLoop
import blocks.serialization

from blocks.select import Selector

import logging
import pprint
logger = logging.getLogger(__name__)

theano.config.floatX='float32'

print theano.config.device

# Dictionaries
import string

all_chars = [ a for a in string.printable]+['<UNK>']
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

if True:
    data_file = 'Shakespeare.poetry.txt'
    dim = 16
    hidden_state_dim = 16
    feedback_dim = 16
else:
    data_file = 'Shakespeare.plays.txt'
    dim = 64
    hidden_state_dim = 64
    feedback_dim = 64
    
seq_len = 256    # The input file is learned in chunks of text this large

# Network parameters
num_states=len(char2code)  # This is the size of the one-hot input and SoftMax output layers

batch_size =  10  # This is for mini-batches : Helps optimize GPU workload
num_epochs =   1  # Number of reads-through of corpus to do first training

data_path = '../data/'  + data_file
save_path = '../models/' + data_file + '.model'

#from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

from fuel.datasets import Dataset

class CharacterTextFile(Dataset):
    provides_sources = ("data",)

    def __init__(self, fname, chunk_len, dictionary, **kwargs):
        self.fname = fname
        self.chunk_len = chunk_len
        self.dictionary = dictionary 
        super(CharacterTextFile, self).__init__(**kwargs)

    def open(self):
        return open(self.fname,'r')

    def get_data(self, state, request):
        assert isinstance(request, int)
        x = numpy.zeros((self.chunk_len, request), dtype='int64')
        for i in range(request):
            txt=state.read(self.chunk_len)
            if len(txt)<self.chunk_len: raise StopIteration
            #print(">%s<\n" % (txt,))
            x[:, i] = [ self.dictionary[c] for c in txt ]
        return (x,)    
    
    def close(self, state):
        state.close()
        
dataset = CharacterTextFile(data_path, chunk_len=seq_len, dictionary=char2code)
data_stream = DataStream(dataset, iteration_scheme=ConstantScheme(batch_size))

a=data_stream.get_data(10)  # i.e. ask for 10 samples of 256(=seq_len) characters
a[0].shape  # (256, 10) - essentially, a 10 separate streams of 256 characters 
#[ code2char[v] for v in [94, 27, 21, 94, 16, 14, 54, 23, 14, 12] ]     # Horizontal stripe in matrix 
#[ code2char[v] for v in [94, 94,95,36,94,47,50,57,40,53,68,54,94,38] ] # Vertical stripe in matrix
''.join([ code2char[v] for v in a[0][:,0] ])  # This a vertical strip - same as markov_chain example

transition = GatedRecurrent(name="transition", dim=hidden_state_dim, activation=Tanh())
generator =  SequenceGenerator(
                Readout(readout_dim=num_states, source_names=["states"],
                        emitter=SoftmaxEmitter(name="emitter"),
                        feedback_brick=LookupFeedback(
                            num_states, feedback_dim, name='feedback'),
                        name="readout"),
                transition,
                weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
                name="generator"
)

generator.push_initialization_config()
transition.weights_init = Orthogonal()
generator.initialize()

print(generator.readout.emitter.readout_dim)

logger.info("Parameters:\n" + pprint.pformat(
    [(key, value.get_value().shape) for key, value in Selector(generator).get_params().items()],
    width=120))

x = tensor.lmatrix('data')
cost = aggregation.mean(generator.cost_matrix(x[:, :]).sum(), x.shape[1])
cost.name = "sequence_log_likelihood"

model=Model(cost)

algorithm = GradientDescent(
    cost=cost, params=list(Selector(generator).get_params().values()),
    step_rule=blocks.algorithms.CompositeRule([blocks.algorithms.StepClipping(10.0), blocks.algorithms.Scale(0.01)]) )  
# tried: blocks.algorithms.Scale(0.001), blocks.algorithms.RMSProp(), blocks.algorithms.AdaGrad()

#  from IPython.display import SVG
#  SVG(theano.printing.pydotprint(cost, return_image=True, format='svg'))
#from IPython.display import Image
#Image(theano.printing.pydotprint(cost, return_image=True, format='png'))

main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=data_stream,
    model=model,
    extensions=[
        FinishAfter(after_n_epochs=num_epochs),
        TrainingDataMonitoring([cost], prefix="this_step", after_batch=True),
        TrainingDataMonitoring([cost], prefix="average",   every_n_batches=100),
        Checkpoint(save_path, every_n_batches=1000),
        Printing(every_n_batches=500)
    ]
)

main_loop.run()

output_length = 1000  # in characters

sampler = ComputationGraph(
    generator.generate(n_steps=output_length, batch_size=1, iterate=True)
)

sample = sampler.get_theano_function()
states, outputs, costs = [data[:, 0] for data in sample()]

numpy.set_printoptions(precision=3, suppress=True)
print("Generation cost:\n{}".format(costs.sum()))

print(''.join([ code2char[c] for c in outputs]))

