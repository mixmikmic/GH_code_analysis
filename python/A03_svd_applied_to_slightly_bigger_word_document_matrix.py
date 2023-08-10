#import pandas for conviently labelled arrays
import pandas
# import numpy for SVD function
import numpy
# import matplotlib.pyplot for visualising arrays
import matplotlib.pyplot as plt

# create a simple word-document matrix as a pandas dataframe, the content values have been normalised
words = ['wheel', ' seat', ' engine', ' slice', ' oven', ' boil', 'door', 'kitchen', 'roof']
print(words)
documents = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9']
word_doc = pandas.DataFrame([[0.5,0.3333, 0.25, 0, 0, 0, 0, 0, 0],
                      [0.25, 0.3333, 0, 0, 0, 0, 0, 0.25, 0],
                      [0.25, 0.3333, 0.75, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0.5, 0.5, 0.6, 0, 0, 0],
                      [0, 0, 0, 0.3333, 0.1667, 0, 0.5, 0, 0],
                      [0, 0, 0, 0.1667, 0.3333, 0.4, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0.25, 0.25],
                      [0, 0, 0, 0, 0, 0, 0.5, 0.25, 0.25],
                      [0, 0, 0, 0, 0, 0, 0, 0.25, 0.5]], index=words, columns=documents)
# and show it
word_doc

# create a numpy array from the pandas dataframe
A = word_doc.values

# break it down into an SVD
U, s, VT = numpy.linalg.svd(A, full_matrices=False)
S = numpy.diag(s)

# what are U, S and V
print("U =\n", numpy.round(U, decimals=2), "\n")
print("S =\n", numpy.round(S, decimals=2), "\n")
print("V^T =\n", numpy.round(VT, decimals=2), "\n")

# rebuild A2 from U.S.V
A2 = numpy.dot(U,numpy.dot(S,VT))
print("A2 =\n", numpy.round(A2, decimals=2))

# S_reduced is the same as S but with only the top 3 elements kept
S_reduced = numpy.zeros_like(S)
# only keep top two eigenvalues
l = 3
S_reduced[:l, :l] = S[:l,:l]
# show S_rediced which has less info than original S
print("S_reduced =\n", numpy.round(S_reduced, decimals=2))

# what is the document matrix now?
S_reduced_VT = numpy.dot(S_reduced, VT)
print("S_reduced_VT = \n", numpy.round(S_reduced_VT, decimals=2))

# plot the array
p = plt.subplot(111)
p.axis('scaled'); p.axis([-2, 2, -2, 2]); p.axhline(y=0, color='lightgrey'); p.axvline(x=0, color='lightgrey')
p.set_yticklabels([]); p.set_xticklabels([])

p.set_title("S_reduced_VT")
p.plot(S_reduced_VT[0,],S_reduced_VT[1,],'ro')

plt.show()

# topics are a linear combination of original words
U_S_reduced = numpy.dot(U, S_reduced)
df = pandas.DataFrame(numpy.round(U_S_reduced, decimals=2), index=words)

# show colour coded so it is easier to see significant word contributions to a topic
df.style.background_gradient(cmap=plt.get_cmap('Blues'), low=0, high=2)

