import tensorflow as tf

a = tf.zeros((3, 3), dtype=tf.float32, name=None)
b = tf.zeros_like(a, dtype=None, name=None)
c = tf.ones((2, 2), dtype=tf.float32, name=None)
d = tf.ones_like(c, dtype=None, name=None)
e = tf.fill((2,1), 3.140, name=None)
f = tf.constant([12,2], dtype=tf.int32, name=None)

sess = tf.Session()
with sess.as_default():
    print 'tf.zeroes'
    print a.eval()
    print 'tf.zeroes_like'
    print b.eval()
    print 'tf.ones'
    print c.eval() 
    print'tf.ones_like'
    print d.eval()
    print 'tf.fill'
    print e.eval()
    print 'tf.constant'
    print f.eval()
sess.close()

#another way of printing TensorFlow is like this 
sess = tf.InteractiveSession()
print sess.run(a)
print sess.run(b)
print sess.run(c)

a = tf.linspace(0.0, 10.0, 10, name="linspace")
b = tf.linspace(0.0, 12.4, 7, name="linspace")
sess = tf.Session()
with sess.as_default():
    print 'tf.linspace'
    print a.eval()
    print b.eval()

# tf.range is for integers 
a = tf.range(4, limit=15, delta=3, name='range')
with sess.as_default():
    print 'tf.range'
    print a.eval()

# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)
with sess.as_default():
    print 'tf.random_normal'
    print norm.eval()

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)
with sess.as_default():
    print 'tf.shuffle'
    print shuff.eval()
    print 'tf.shuffle second time'
    print shuff.eval()

a= tf.random_normal((2,6), mean=0.2, stddev=1.0, seed=None, name='Norm')
print sess.run(a)
print sess.run(a)

tf.set_random_seed(1234)
a = tf.random_normal((1,1), mean=0.2, stddev=1.0, name='Norm')
with sess.as_default():
    print a.eval()
    print a.eval()
with tf.Session() as sess1:
    print a.eval()
    print a.eval()

tf.set_random_seed(1234)
a = tf.random_uniform([1], seed=4321)
b = tf.random_normal([1],  seed=4321)

# Repeatedly running this block with the same graph will generate different
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'

#tf.set_random_seed(1234)
a = tf.random_uniform([1], seed= 1234)
b = tf.random_normal([1])

# Repeatedly running this block with the same graph will generate different
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'



