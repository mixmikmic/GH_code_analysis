from utils import plot_xy
from matplotlib import pyplot as plt

datadir = '../data'
num_pass = 1
batchsize = 32
stepsize = 1

import medlda

topics = [5, 10, 20, 25, 30, 35, 40, 45, 50, 100, 200]
accs = []
for num_topic in [25, 30, 35, 40, 45]:
    pamedlda = medlda.OnlineGibbsMedLDA(num_topic=num_topic, labels = 2,
                                    words = 61188, stepsize=stepsize)

    for pi in range(num_pass):
        pamedlda.train_with_gml('%s/binary_train.gml' % datadir, batchsize)
        
    (pred, ind, acc) = pamedlda.infer_with_gml('%s/binary_test.gml' % datadir, 100)
    
    accs.append(acc)
    

result = {}
topics = [5, 10, 20, 25, 30, 35, 40, 45, 50, 100, 200]
result['baseline'] = [0.8295254833040422,
0.827768014059754,
0.81195079086116,
0.81195079086116,
0.7768014059753954,
0.7311072056239016,
0.5746924428822495,
0.5588752196836555,
0.5588752196836555,
0.5588752196836555,
0.5588752196836555
]

plt.plot(topics, result['baseline'], marker='o')
plt.xlabel('number of topics')
plt.ylabel('accuracy')
plt.savefig('plot_degradation_topic.pdf')

datadir = '../data'
num_pass = 1
batchsize = 32
stepsize = 10

import medlda

topics = [5, 10, 20, 25, 30, 35, 40, 45, 50, 100, 200]
accs = []
for num_topic in topics:
    pamedlda = medlda.OnlineGibbsMedLDA(num_topic=num_topic, labels = 2,
                                    words = 61188, stepsize=stepsize)

    for pi in range(num_pass):
        pamedlda.train_with_gml('%s/binary_train.gml' % datadir, batchsize)
        
    (pred, ind, acc) = pamedlda.infer_with_gml('%s/binary_test.gml' % datadir, 100)
    
    accs.append(acc)
    

result['stepsize'] = [0.827768,
                      0.827768,
                      0.8400703,
                      0.8154657,
                      0.8260105,
                      0.8295255,
                      0.827768,
                      0.8242531,
                      0.8330404,
                      0.8189807,
                      0.7996586,
]



plot_xy([topics, topics], [result['baseline'], result['stepsize']], 
        names=['original', 'ghost-copy'], xlabel='number of topics', ylabel='accuracy')

datadir = '../data'
num_pass = 5
batchsize = 32
stepsize = 1

import medlda

topics = [5, 10, 20, 25, 30, 35, 40, 45, 50, 100, 200]
accs = []
for num_topic in topics:
    pamedlda = medlda.OnlineGibbsMedLDA(num_topic=num_topic, labels = 2,
                                    words = 61188, stepsize=stepsize)

    for pi in range(num_pass):
        pamedlda.train_with_gml('%s/binary_train.gml' % datadir, batchsize)
        
    (pred, ind, acc) = pamedlda.infer_with_gml('%s/binary_test.gml' % datadir, 100)
    
    accs.append(acc)
    

result['pass'] = accs



plt.plot(topics, result['baseline'], marker='o', color='r')
plt.plot(topics, result['stepsize'], marker='+', color='g')
plt.plot(topics, result['pass'], marker='x', color='b')
plt.xlabel('number of topics')
plt.ylabel('accuracy')
plt.savefig('plot_solve_degradation.pdf')

result['pass'] = [0.8295255,
                  0.8207381,
                  0.8330404,
                  0.8383128,
                  0.8295255,
                  0.8383128,
                  0.8400703,
                  0.8365554,
                  0.8383128,
                  0.8435852,
                  0.5588752
]

plot_xy([topics, topics, topics], [result['baseline'], result['stepsize'], result['pass']], 
        names=['original', 'ghost copies', 'more passes'], xlabel='number of topics', ylabel='accuracy')

result



