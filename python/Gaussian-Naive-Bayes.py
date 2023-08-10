from collections import defaultdict
import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.trained = defaultdict(bool)
        self.stats = defaultdict(dict)
        
    def _update_stats(self, datum, label):
        if len(datum) > len(self.stats[label]):
            self.stats[label]['n'] = 1
            self.stats[label]['sum'] = datum
            self.stats[label]['sum_squared'] = datum ** 2
            self.stats[label]['mean'] = datum
            self.stats[label]['std'] = np.zeros(len(datum))
        else:
            stats = self.stats[label]
            self.stats[label]['n'] += 1
            self.stats[label]['sum'] += datum
            self.stats[label]['sum_squared'] += datum ** 2
            self.stats[label]['mean'] =  stats['sum'] / stats['n']
            self.stats[label]['std'] = (stats['sum_squared'] - (stats['sum'] ** 2 / stats['n'])) / stats['n']
            self.trained[label] = True
        
    def train_incremental(self, data, labels):
        data_array = np.array(data)
        labels_array = np.array(labels)
        
        for datum, label in zip(data_array, labels_array):
            self._update_stats(datum, label)    
                
    def train_batch(self, data, labels):
        pass
    
    def _gaussian(self, x, mu, sigma):
        num = (x - mu) ** 2
        denum = 2 * sigma ** 2
        norm = 1 / np.sqrt(2 * np.pi * sigma ** 2)
        return norm * np.exp(-num / denum)
        
    def predict(self, data):
        if len(data) == 1:
            data = [data]
            
        output = []
        
        for datum in data:
            best = (-1, None)
            for label in self.stats.keys():
                if self.trained[label]:
                    value = self._gaussian(np.array(datum), self.stats[label]['mean'], self.stats[label]['std'])
                    likelihood = np.nanprod(value)
                    if likelihood > best[0]:
                        best = (likelihood, label)
                else:
                    print('Not even training data for {}'.format(label))
                
            output.append(best[1])    
                
        return output

data = [
    [1, 2, 3, 4],
    [10, 20, 30, 40],
    [1.2, 1.9, 3.1, 4.5],
    [10.1, 20.2, 29.8, 42]
]

labels = ['x', 'y', 'x', 'y']

nb = GaussianNaiveBayes()

nb.train_incremental(data, labels)

nb.stats

nb.predict(data)

nb.train_incremental([[100, 200, 300, 400]], ['z'])

nb.stats

nb.predict([[101, 198, 305, 401]])

nb.train_incremental([[103, 205, 299, 412]], ['z'])

nb.predict([[101, 198, 305, 401]])

nb.predict([[1,2,3,4]])

nb.stats

