# import numpy for SVD function
import numpy
# import matplotlib.pyplot for visualising arrays
import matplotlib.pyplot as plt

# create a really simple matrix
A = numpy.array([[-1,1], [1,1]])
# and show it
print("A = \n", A)

# plot the array
p = plt.subplot(111)
p.axis('scaled'); p.axis([-2, 2, -2, 2]); p.axhline(y=0, color='lightgrey'); p.axvline(x=0, color='lightgrey')
p.set_yticklabels([]); p.set_xticklabels([])

p.set_title("A")
p.plot(A[0,],A[1,],'ro')

plt.show()

# break it down into an SVD
U, s, VT = numpy.linalg.svd(A, full_matrices=False)
S = numpy.diag(s)

# what are U, S and V
print("U =\n", U, "\n")
print("S =\n", S, "\n")
print("V^T =\n", VT, "\n")

for px in [(131,U, "U"), (132,S, "S"), (133,VT, "VT")]:
    subplot = px[0]
    matrix = px[1]
    matrix_name = px[2]
    p = plt.subplot(subplot)
    
    p.axis('scaled'); p.axis([-2, 2, -2, 2]); p.axhline(y=0, color='lightgrey'); p.axvline(x=0, color='lightgrey')
    p.set_yticklabels([]); p.set_xticklabels([])

    p.set_title(matrix_name)
    p.plot(matrix[0,],matrix[1,],'ro')
    pass

plt.show()

# rebuild A2 from U.S.V
A2 = numpy.dot(U,numpy.dot(S,VT))
print("A2 = \n", A2)

# plot the reconstructed A2
p = plt.subplot(111)
p.axis('scaled'); p.axis([-2, 2, -2, 2]); p.axhline(y=0, color='lightgrey'); p.axvline(x=0, color='lightgrey')
p.set_yticklabels([]); p.set_xticklabels([])

p.set_title("A2")
p.plot(A2[0,],A2[1,],'ro')

plt.show()

