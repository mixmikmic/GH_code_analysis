import tensorflow as tf

init_op = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001, allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init_op)

params = tf.constant([10, 20, 30, 40], dtype=tf.int32, name='params')
ids = tf.constant([0, 0, 2])

print('embedding_lookup(params, ids):\t', tf.nn.embedding_lookup(params, ids).eval())
print('tf.gather(params, ids):       \t', tf.gather(params, ids).eval())

params1 = tf.constant([100,200])
params2 = tf.constant([300,400])
ids = tf.constant([0, 1, 2, 3, 0])
print(tf.nn.embedding_lookup([params1, params2], ids).eval())

