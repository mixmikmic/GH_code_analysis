from bayespy.nodes import *
from bayespy.utils import misc
def MappedCategoricalMixture(thetas, indices, p, **kwargs):
    return MultiMixture(thetas, Categorical, Take(p, indices), **kwargs)

lambda1 = Dirichlet([5,20])
lambda2 = Dirichlet([[20,5], # if lambda1=False
                     [5,20]]) # if lambda1=True

lambda5 = Dirichlet([[20,5], # none (theta1,theta2) are True
                     [12.5,12.5],# exactly one of (theta1,theta2) is True
                     [5,20]]) # both of (theta1,theta2) are True

numStudents = 11
theta1 = Categorical(lambda1, plates=(numStudents,))
theta2 = Mixture(theta1, Categorical, lambda2)
theta5 = MappedCategoricalMixture([theta1, theta2],
                                  [[0, 1], [1, 2]],
                                  lambda5)

