#let's start by importing tensorflow
import tensorflow as tf

#our input x is a float, of tensor dimension [None,1]. The None specifies that we might pass any number of input
#points to x, while the 1 specifies that x is a scalar (i.e. 1 dimensional)
x = tf.placeholder(tf.float32, [None,1])
#each of our variables are initialized to a random number 
a=tf.Variable(tf.random_normal([1],stddev=40.0))
b=tf.Variable(tf.random_normal([1],stddev=40.0))
c=tf.Variable(tf.random_normal([1],stddev=40.0))
#y is like our y=f(x): we tell Tensorflow how to combine a,b,c,x to get f(x)
y=tf.add(tf.add(tf.multiply(a,tf.multiply(x,x)),tf.multiply(b,x)),c)
#Note: instead of the above lengthly expression, we can use the shorthand y=a*x*x+b*x+c


#let's run this function on input [1],[2],[3],[4]: that is, we want to compute f([1]), f([2]),.... 
#first, we need to initialize all our variables and run a session. Why all the extra code? 
#This is because Tensorflow doesn't actually run the model in Python (python is just a front-end); this is so that
#your code can run faster

X=[[1],[2],[3],[4]]
Target=[[-1],[2],[3],[4]]
#code we have to call every time
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#let's get our ouput
output=sess.run(y,feed_dict={x: X})
print(output)





#Write a function to plot output and Target on the same axis



#This probably isn't a very good fit, so let's train our model
#First, we need to define an error function. Let's start with the mean squared error

#We define an error function in exactly the same way we defined f(x) above. Our error function takes 
#y=the output of f(x), and a new variable y_=the target value we want y to be. 
#First, we initialize a placeholder y_ to hold our target answers (just like we initialized x before)
y_=tf.placeholder(tf.float32,[None,1])
#define the error function
error=tf.reduce_mean(tf.multiply(tf.subtract(y_, y),tf.subtract(y_,y)))
#the above is equivalent to the average of (y_-y)^2

#Here's the current error on the dataset
print(sess.run(error,feed_dict={x: X,y_:Target}))



#now let's try to reduce that error

train_step = tf.train.GradientDescentOptimizer(5).minimize(error)
for i in range(1):
    sess.run(train_step, feed_dict={x: X,y_:Target})
#print the current values of (a,b,c)
print("a is", sess.run(a))
print("b is", sess.run(b))
print("c is", sess.run(c))
    
#Now write functions to get the current error of the model, and plot the graph of the model and target on the data





#let's create a model which takes a 10 dimensional vector v as input, and returns soft_max(vM+b)
input_v=tf.placeholder(tf.float32,[None,10])
MatrixVariable=tf.Variable(tf.random_normal([10,10],stddev=40.0))
b_variable=tf.Variable(tf.random_normal([10],stddev=40.0))
output_vector=tf.nn.softmax(tf.matmul(input_v,MatrixVariable) + b_variable)

#sample input
sample_input=[[1,2,3,4,5,6,7,8,9,10]]

#run the model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
output=sess.run(output_vector,feed_dict={input_v: sample_input})
print(output)
sess.close()

##########code your data generation process






#########code your model


#########code your error function



########code your training procedure



#######test your model



