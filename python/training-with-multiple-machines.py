from mxnet import kv, nd
store = kv.create('local')
shape = (2, 3)
x = nd.random_uniform(shape=shape)
store.init('weight', x) 
print('=== init "weight" ==={}'.format(x))

from mxnet import gpu
ctx = [gpu(0), gpu(1)]
y = [nd.zeros(shape, ctx=c) for c in ctx]
store.pull('weight', out=y)
print('=== pull "weight" to {} ===\n{}'.format(ctx, y))

z = [nd.ones(shape, ctx=ctx[i])+i for i in range(len(ctx))]
store.push('weight', z)
print('=== push to "weight" ===\n{}'.format(z))
store.pull('weight', out=y)
print('=== pull "weight" ===\n{}'.format(y))

print('total number of workers: %d'%(store.num_workers))
print('my rank among workers: %d'%(store.rank))

