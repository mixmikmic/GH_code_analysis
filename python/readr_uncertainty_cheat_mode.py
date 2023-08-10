import numpy as np
import matplotlib.pyplot as plt

from puzzle_dont_open import Puzzle

puzzle=Puzzle()

MySequence = [1,2,4]
puzzle.test_sequence(MySequence)

def my_rule(Sequence):
    # Write a function that returns True or False if the sequence follows the rule
    return Sequence[0] == 1

puzzle.test_rule(my_rule)

def uniform_sample(xmin, xmax, ymin, ymax):
    return np.random.random(2,)*np.array([xmax-xmin,ymax-ymin])+np.array([xmin, ymin])

nsamples = 1000
samples = np.zeros([nsamples,2])
for i in range(nsamples):
    samples[i]=uniform_sample(-10, 10, 0, 5)

plt.scatter(samples[:,0],samples[:,1])
plt.show()

def sample_randv(pdf, xmin, xmax, ymin, ymax):
    while True:
        sample = uniform_sample(xmin, xmax, ymin, ymax)
        if pdf(sample[0])>sample[1]:
            break
    
    return sample[0]

norm_pdf = lambda x: np.exp(-x**2/2)

NormSamples = []
for i in range(10000):
    NormSamples.append(sample_randv(norm_pdf, -10, 10, 0, 1))

print('Mean: {}'.format(np.mean(NormSamples)))
print('Variance: {}'.format(np.var(NormSamples)))

plt.hist(NormSamples, bins='auto')
plt.show()

YSamples = np.array(NormSamples)**2

fYDerived = lambda x: np.exp(-x/2)/np.sqrt(2*np.pi*x)
    
print('Mean: {}'.format(np.mean(YSamples)))
print('Variance: {}'.format(np.var(YSamples)))

nbins = 100

xrange = np.arange(0.1,10,0.01)
plt.plot(xrange, 14*nbins*fYDerived(xrange))

plt.hist(YSamples, bins=nbins)
plt.show()

YSamples = 0.2*np.array(NormSamples)+3
    
print('Mean: {}'.format(np.mean(YSamples)))
print('Variance: {}'.format(np.var(YSamples)))

plt.hist(YSamples, bins='auto')
plt.show()

def f1(x):
    return norm_pdf(x)

def f2(x):
    if x>1 or x<-1:
        return 0
    else:
        return 1
    
def f3(x):
    y = 1-np.abs(x)
    if y<0:
        return 0
    else:
        return y
    
def f4(x):
    y = np.abs(x)
    if y>1:
        return 0
    else:
        return y
    
def f5(x):
    y = np.cos(x)
    if y<0 or x>np.pi or x<-np.pi:
        return 0
    else:
        return y

samples = []
for i in range(10000):
    samples.append(sample_randv(f4, -10, 10, 0, 1))

print('Mean: {}'.format(np.mean(samples)))
print('Variance: {}'.format(np.var(samples)))

plt.hist(samples, bins='auto')
plt.show()

funcs = [f1,f2,f3,f4,f5]

NRandomFunctions = 20
NSamples = 2000

samples = []
for i in range(NRandomFunctions):
    fsamples = []
    fWeight = 2*np.random.random()
    iFunction = np.random.randint(len(funcs))
    for j in range(NSamples):
        fsamples.append(fWeight*sample_randv(funcs[iFunction], -10, 10, 0, 1))
    samples.append(fsamples)
    
sampleArray = np.array(samples)
samples = sampleArray.mean(axis=0)

print('Mean: {}'.format(np.mean(samples)))
print('Variance: {}'.format(np.var(samples)))

xrange = np.arange(-0.5,0.5,0.01)

plt.hist(samples, bins='auto')
plt.plot(xrange,160*np.exp(-xrange**2/(2*np.var(samples))))
plt.show()



