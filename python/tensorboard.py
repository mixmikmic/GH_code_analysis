# Import Dependencies
import tensorflow as tf

with tf.name_scope("OPERATION_A"):
    a = tf.add(1,2, name="First_add")
    a1 = tf.add(100,200, name='a_add')
    a2 = tf.multiply(a,a1)

with tf.name_scope("OPERATION_B"):
    b = tf.add(3,4, name="Second_add")
    b1 = tf.add(300,400, name='b_add')
    b2 = tf.multiply(b,b1)

c = tf.multiply(a2,b2, name="Final_result")

with tf.Session() as sess:
    # Save Current Session Graph
    writer = tf.summary.FileWriter('./output', sess.graph)
    print(sess.run(c))
    writer.close

k = tf.placeholder(tf.float32)

mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k),stddev=1)
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./tmp/histogram_example")
    summaries = tf.summary.merge_all()

    N=400
    for step in range(N):
        k_val = step/float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)
    writer.close()

