# Import Tensorflow
import tensorflow as tf

print(tf.__version__)

# Defining Contants in TF
hello = tf.constant("Hello")

world = tf.constant("World")

type(hello)

type(world)

print(hello)

# Run the Operation inside the session
# "with": Use it as it automatically opens the session and closes it at the end. So no need to specifically close the session.
with tf.Session() as sess:
    # Run the block of Code
    result = sess.run(hello+world)

print(result)

# Addition Example
a = tf.constant(20)

b = tf.constant(12)

a + b

a + b

with tf.Session() as sess:
    result = sess.run(a+b)

result

# Other Operations
const = tf.constant(15)

# Filled Matrix. ex. 3 x 3 matrix filled with value 12
fill_mat = tf.fill((3,3),12)

# Tensor with all Zeros
# ex. 4x4 matrix of zeros
zero = tf.zeros((4,4))

# Tensor with all Ones
# ex. 2x2 matrix of 1's
one = tf.ones((2,2))

# Random Normal Distribution
# Returns random values from a normal distribution
# Inputs: shape, mean, stddev, seed, name.
randn = tf.random_normal((4,4), mean=0, stddev=0.1)

# Uniform Distribution
# Inputs: shape, minval, maxval, seed, name
randu = tf.random_uniform((4,4), minval=0, maxval=2)

# Put all operations into a List
my_ops = [const, fill_mat, zero, one, randn, randu]

# New: TF Interactive Session
# Use: Only in Jupyter Notebook Env.
# All lines after this are considered to be in the 
# with tf.Session() 
sess = tf.InteractiveSession()

for op in my_ops:
    print(sess.run(op))
    print('\n')

# Most of the functions come with eval method
for op in my_ops:
    print(op.eval())
    print('\n')

# Matrix Multiplication
a = tf.constant([[1,2],
                [3,4]])

# Shape of Matrix "a": 2 x 2
a.get_shape()

b = tf.constant([[10],
                 [100]])

# Shape of Matrix "b" : 2 x 1
b.get_shape()

# Result of Matrix Multiplication
result = tf.matmul(a,b)

# Evaluate the Matrix Multiplication
sess.run(result)

