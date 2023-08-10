from bayespy.nodes import Categorical, Beta, Mixture
lambda1 = Beta([20,5]) 
lambda2 = Beta([[5,20],[20,5]])
lambda3 = Beta([ [[5,20],[20,5]],
                 [[3,40],[40,3]] ])
theta1 = Categorical(lambda1)
theta2 = Mixture(theta1, Categorical, lambda2)
theta3 = Mixture(theta1, Mixture, theta2, Categorical, lambda3)

pi1 = Beta([[[5,20], [20,5]]], plates=(10,2))
pi2 = Beta([[[5,20], [20,5]]], plates=(10,2))
pi3 = Beta([[[5,20], [20,5]]], plates=(10,2))

from bayespy.nodes import Bernoulli
X1 = Mixture(theta1, Bernoulli, pi1)
X2 = Mixture(theta2, Bernoulli, pi2)
X3 = Mixture(theta3, Bernoulli, pi3)

X1.observe([1,1,1,1,1,1,1,1,1,1])
X2.observe([0,0,0,0,0,0,0,0,0,0])
X3.observe([0,1,0,1,0,1,0,1,0,1])

from bayespy.inference import VB
Q = VB(X1, X2, X3, pi1, pi2, pi3, theta3, theta2, theta1, lambda1, lambda2, lambda3)
Q.update(repeat=100)

print(theta1.get_moments()[0])
print(theta2.get_moments()[0])
print(theta3.get_moments()[0])

