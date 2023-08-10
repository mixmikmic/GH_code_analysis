import tensorflow as tf
import os

# Saved variables
#
# var1 = tf.Variable(2., name='var1'); print('var1:', var1)
# var2 = tf.Variable(5., name='var2'); print('var2:', var2)
# var3 = tf.add(var1, var2, name='add'); print('var3:', var3)
#
# Saved files
#
# ./save/all_var.ckpt.data-00000-of-00001
# ./save/all_var.ckpt.index
# ./save/all_var.ckpt.meta

# Restore
saved_data = './save/all_var.ckpt'
meta_data = saved_data + '.meta'
saver = tf.train.import_meta_graph(meta_data)
graph = tf.get_default_graph()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, saved_data) # Load

    # Check variables
    for v in tf.global_variables():
        print(v.name)
    var1_restored = graph.get_tensor_by_name('var1:0')
    var2_restored = graph.get_tensor_by_name('var2:0')
    var3_restored = graph.get_tensor_by_name('add:0') # var3 is not a variable
    
    print('var1 =', sess.run(var1_restored))
    print('var2 =', sess.run(var2_restored))
    print('var3 =', sess.run(var3_restored))

