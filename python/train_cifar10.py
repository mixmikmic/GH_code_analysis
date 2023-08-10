import mxnet
import mxnet as mx
import train_cifar10

# Set up the hyper-parameters
args = train_cifar10.command_line_args(defaults=True)
args.gpus = "0"
#args.network = "lenet"  # Fast, not very accurate
#args.network = "inception-bn-28-small"  # Much more accurate & slow

# Configure charts to plot while training
from mxnet.notebook.callback import LiveLearningCurve
cb_args = LiveLearningCurve('accuracy', 5).callback_args()

# Start training
train_cifar10.do_train(args, 
    callback_args=cb_args,
)



