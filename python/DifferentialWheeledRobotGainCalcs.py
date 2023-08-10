import numpy as np



#  USER INPUT FOR GAINS:
kp = 1
ka = 2
kb = -1

lin_sys = np.matrix([[-kp, 0, 0],[0, -(ka-kp), -kb],[0, -kp, 0]])

eigen = np.linalg.eig(lin_sys)
print (eigen[0])

