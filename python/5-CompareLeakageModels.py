import numpy as np
import matplotlib.pyplot as plt

# 8-component leakage function; beta is a 9-element array
def leakage9(x, beta):
    result = beta[8]
    for i in range(0, 8):
        bit = (x >> i) & 1  # this is the definition: gi = [bit i of x]
        result += beta[i] * bit
    return result

# Hamming weight leakage model
def leakageHWpure(x, beta):
    return byteHammingWeight[x]

# Hamming weight leakage model with coefficient and intercept obtained by linear regression
# beta is a 2-element array
def leakageHWfitted(x, beta):
    return beta[0] * byteHammingWeight[x] + beta[1]

byteHammingWeight = np.load('../data/bytehammingweight.npy') # HW model
beta9 = np.load('results/lrmodel9_1000traces.npy')           # LR model with 9 coefs
betaHW = np.load('results/lrmodelhw.npy')                    # LR model with coefs
predictionsT = np.load('results/means2000.npy')              # Templates (reduced)

predictions9 = np.zeros(256)
predictionsHW = np.zeros(256)

for x in range(256):
    predictions9[x] = leakage9(x, beta9)
    predictionsHW[x] = leakageHWfitted(x, betaHW) # replace by leakageHWpure to see no difference

c1 = np.corrcoef(predictionsHW, predictions9)
c2 = np.corrcoef(predictionsHW, predictionsT)
c3 = np.corrcoef(predictions9, predictionsT)
print("Correlation HW to LR9: %.02f" % c1[0,1])
print("Correlation HW to T  : %.02f" % c2[0,1])
print("Correlation LR9 to T : %.02f" % c3[0,1])

for x in range(256):
    line = np.array([predictions9[x], predictionsHW[x]])
    plt.plot(np.array([x, x]), line, '-', color='silver')
p1, = plt.plot(predictions9, 'r.')  # left-hand size is for later use in the legend
p2, = plt.plot(predictionsHW, 'g.')

plt.xlim(-1, 256)
plt.xlabel('Value of the intermediate variable')
plt.ylabel('Leakage predicted by a model')
plt.title('Correlation: %f' % c1[0,1])
plt.legend([p1, p2], ['9-component LR fitted', 'Hamming weight'], loc='best', numpoints=1)
plt.show()

for x in range(256):
    line = np.array([predictions9[x], predictionsT[x]])
    plt.plot(np.array([x, x]), line, '-', color='silver')
p1, = plt.plot(predictions9, 'r.')  # left-hand size is for later use in the legend
p3, = plt.plot(predictionsT, 'b.')

plt.xlim(-1, 256)
plt.xlabel('Value of the intermediate variable')
plt.ylabel('Leakage predicted by a model')
plt.title('Correlation: %f' % c3[0,1])
plt.legend([p1, p3], ['9-component LR fitted', 'Templates'], loc='best', numpoints=1)
plt.show()

