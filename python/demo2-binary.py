import mxnet as mx
from movielens_data import get_data_iter, max_id
from matrix_fact import train
import recotools

# If MXNet is not compiled with GPU support (e.g. on OSX), set to [mx.cpu(0)]
# Can be changed to [mx.gpu(0), mx.gpu(1), ..., mx.gpu(N-1)] if there are N GPUs
ctx = [mx.gpu(0)]

pos_train_data, pos_test_data = get_data_iter(batch_size=100)
max_user, max_item = max_id('./ml-100k/u.data')
(max_user, max_item)

train_data = recotools.NegativeSamplingDataIter(pos_train_data, sample_ratio=3, positive_label=0, negative_label=1)
test_data = recotools.NegativeSamplingDataIter(pos_test_data, sample_ratio=3,   positive_label=0, negative_label=1)
train_test_data = (train_data, test_data)

def plain_net(k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    label = mx.symbol.Variable('score')
    # user feature lookup
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    # item feature lookup
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    # loss layer
    pred = recotools.CosineLoss(a=user, b=item, label=label)
    return pred

net1 = plain_net(64)
mx.viz.plot_network(net1)

results1 = train(net1, train_test_data, num_epoch=20, learning_rate=0.02, ctx=ctx)



