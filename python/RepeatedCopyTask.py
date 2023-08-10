from util import *
from dnc_lstm import DNC
from dnc_ff import DNCFF
from autograd import grad
from autograd.misc.optimizers import rmsprop, adam

def repeat_copy(max_seq_len, seq_width, max_repeat=4):
    """
    Implements the repeated copy task
    max_seq_len - maximum length of the sequence
    seq_width - size of bit vector at each time step
    max_repeat - maximum number of repetitions
    """
    seq_len = 1 + np.random.randint(max_seq_len)
    repeat = 1 + np.random.randint(max_repeat)
    rand_tile = np.random.randint(2, size=(seq_len, seq_width))
    inputs = np.zeros((seq_len*(1+repeat)+3, seq_width+2))
    inputs[0,1] = 1
    inputs[seq_len+1,0] = repeat
    inputs[1:seq_len+1, 2:] = rand_tile
    targets = np.zeros((seq_len*(1+repeat)+3, seq_width+1))
    targets[-1,0] = 1
    targets[seq_len+2:-1, 1:] = np.vstack([rand_tile]*repeat)
    mask = np.ones_like(targets)
#     mask = np.zeros((seq_len*2+3, seq_width+1))
#     mask[seq_len+2:,:] = 1
    return inputs, targets, mask

inputs, targets, mask = repeat_copy(4, 4)
display(inputs.T)
display(targets.T)
display(mask.T)

## Testing recurrent DNC

def loss_fn(pred, target, mask):
    pred = sigmoid(pred)
    one = np.ones_like(pred)
    epsilon = 1.e-20 # to prevent log(0)
    a = target * np.log(pred + epsilon)
    b = (one - target) * np.log(one - pred + epsilon)
    return np.mean(- (a + b) * mask)

seq_len, seq_wid = 4, 4

# dnc = DNCFF(input_size=seq_wid+2, output_size=seq_wid+1, hidden_size=32, R=2, N=64, W=4)
dnc = DNC(input_size=seq_wid+2, output_size=seq_wid+1, hidden_size=32, R=2, N=64, W=8)
dnc_params = dnc._init_params()

def print_training_prediction(params, iters):
    
    inputs, targets, mask = repeat_copy(seq_len, seq_wid)
    result = []
    dnc = DNC(input_size=seq_wid+2, output_size=seq_wid+1, hidden_size=32, R=2, N=64, W=8)
    for t in range(inputs.shape[0]):
        out = dnc.step_forward(params, inputs[np.newaxis, t])
        result.append(out)
    result = np.concatenate(result, axis=0)
    loss = loss_fn(result, targets, mask)
    print "Test loss: ", loss
    print "Input"
    display(inputs.T)
    print "Target"
    display(targets.T)
    print "Predicted"
    display((sigmoid(result)).T)
    display(np.around((sigmoid(result) * mask), decimals=0).astype('int').T)
    
    # Saving Model Check Points
    save_pickle(params, './ckpt/repeated_copy/Iter_%d_Loss_%.6f.pkl' % (iters, loss))

    
def training_loss(params, iters):
    inputs, targets, mask = repeat_copy(seq_len, seq_wid)
    result = []
    dnc = DNC(input_size=seq_wid+2, output_size=seq_wid+1, hidden_size=32, R=2, N=64, W=8)
    for t in range(inputs.shape[0]):
        out = dnc.step_forward(params, inputs[np.newaxis, t])
        result.append(out)
    result = np.concatenate(result, axis=0)
    return loss_fn(result, targets, mask)

def callback(weights, iters, gradient):
    if iters % 1000 == 0:
        print("Iteration", iters, "Train loss:", training_loss(weights, 0))
        print_training_prediction(weights, iters)

# Build gradient of loss function using autograd.
training_loss_grad = grad(training_loss)

print("Training DNC...")
# trained_params = adam(training_loss_grad, dnc_params, step_size=0.001,
#                       num_iters=100000, callback=callback)
trained_params = rmsprop(training_loss_grad, dnc_params, step_size=0.001,
                      num_iters=100000, callback=callback)



