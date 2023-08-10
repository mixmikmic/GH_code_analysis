import tensorflow as tf # This gives access to tensorflow libraries. Note that  Tensorflow was installed in a python2.7  
                        # environment in anaconda

helloworld = tf.constant('Hello, world!') # constant varirable declaration

with tf.Session() as sess:      # create a session
    print(sess.run(helloworld)) #print output of running the the constant c in a ssession

helloworld = tf.constant("hello world");
constant_Value = tf.constant(5) # Scalar. Note absence of square brackets. This is a Scalar Constant
  

#Note the shape . A scalar constant indeed!
print(constant_Value) # we can also use print(x.get_shape)

#Lets change the constant:
constant_Value_2 =  tf.constant([5])
constant_value_3 = tf.constant([5,4,5])
print(constant_Value_2.shape) # we can use consatant_value_2.shape too as illutrated here.
print(constant_Value_2)      
print(constant_Value_2.get_shape)# Note the constant is now a 1 d - that is a vector
print(constant_value_3)# the shape is still 1 d - a vector. Observe the same thing for examples on variables.
                       #Constant_value_2 and _3 are of same dimension. See note on variable on how to 
                       # read this

constant_Value_4 =  tf.constant([[5,3,5],[2,3,5]])
print(constant_Value_4)      
print(constant_Value_4.get_shape)# Note the constant is now a 2 d - matrix

y = tf.Variable([5]) 
#Or you can create the variable first and assign a value later 
#like this (note that you always have to specify a default value):
y = tf.Variable([0]) # specify default value
y = y.assign([10])

# Now look at the examplw below. The first print shows that y is a tensfor flow object 
# (its type) and it still not a tensorflow program until we run it in a session. 
# when run in a session so that  we see that the value we expected was returned.  
# we might expect - the value we assigned.

print(y)
with tf.Session() as sess:     
    print(sess.run(y))

y = tf.Variable([2,3]) # specify default value
y = y.assign([2,3])
print(y)
with tf.Session() as sess:     
    print(sess.run(y))
    

y = tf.Variable([[2,3],[2,1]]) # specify default value
y = y.assign([[2,3],[4,5]])
print(y)
with tf.Session() as sess:     
    print(sess.run(y))

y = tf.Variable([[2,3,4],[2,1,2]]) # specify default value
y = y.assign([[2,3,4],[4,5,3]])
print(y)
with tf.Session() as sess:     
    print(sess.run(y))





