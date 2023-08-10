import numpy as np
from sklearn.decomposition import NMF

K = 3
model = NMF(n_components=K,init='nndsvdar',solver='mu') 
model

Original = [
    [5,3,0,1],
    [4,0,0,1],
    [1,1,0,0],
    [1,0,0,1],
    [0,1,5,0]
]
Original = np.array(Original)

W = model.fit_transform(Original)
H = model.components_

print("W")
print(W)
print("H")
print(H)

crossValue = np.dot(W,H)
print("crossValue \n",crossValue)
print("rounded Values\n",np.round(crossValue))
print("Original\n",Original)

import matplotlib.pyplot as plt
def plotCompare(Original,prediction):
    N = Original.shape[0]
    last = Original.shape[1]-1
    ind = np.arange(N)  # the x locations for the groups
    width = 0.17       # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, Original[:,last], width, color='r')
    rects2 = ax.bar(ind + width, prediction[:,last], width, color='b')
    rects3 = ax.bar(ind + width+width, np.round(prediction[:,last],2), width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Death Probability')
    ax.set_title('Comparation of Values')
    ax.set_xticks(ind+ width / last)
    ax.set_xticklabels(('G1', 'G2', 'G3', 'G4','G5','G6'))

    ax.legend((rects1[0], rects2[0], rects3[0]), ('Original', 'Cross Value','Round Cross Value'))

    plt.show()

plotCompare(Original,crossValue)

def matrix_factorization(R, K = 2, steps=5000, alpha=0.0002, beta=0.02,error = 0.001):
    N = len(R)
    M = len(R[0])
    P = np.random.rand(N,K)
    Q = np.random.rand(K,M)
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
#        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < error:
            break
    return P, Q

nP, nQ = matrix_factorization(Original,K)

nP

nQ

prediction = np.dot(nP,nQ)
print(prediction)

np.around(prediction,2)

Original

plotCompare(Original,prediction)



