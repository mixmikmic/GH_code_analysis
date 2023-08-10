import numpy
S4 = numpy.matrix([[-1,-1, 1,   0],  
                  [-2 ,0,  1.5, 2],  # H balance
                  [-1,-2,  1,  1],   # O balance
                  [1,  0,   0, 0]])

C = numpy.matrix([[0,0,0,1]]).T
Y = numpy.linalg.solve(S4,C)
print(Y)

import numpy
S3 = numpy.matrix([[-1,-1, 1],     
                  [-4 ,0, 3.5],   # DOR balance
                  [1,  0,   0]])

C = numpy.matrix([[0,0,1]]).T
Y = numpy.linalg.solve(S3,C)
print(Y)



