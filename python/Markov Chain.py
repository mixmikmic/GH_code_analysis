import numpy as np

class Markov:
    def __init__ (self, order):
        self.order = order
        self.group_size = self.order + 1
        self.text = None
        self.graph = {}
        return

    def fit(self, filename): 
        self.text = open(filename).read().split() 
        self.text = self.text + self.text [:self.order]

        for i in range (0, len (self.text) - self.group_size):
            key = tuple (self.text [i : i + self.order] ) 
            value = self.text [i + self.order]

            if key in self.graph:
                self.graph[key].append (value)
            else:
                self.graph[key] = [value]    
        return

    def generate (self,length):
        index = np.random.randint(0, len(self.text) - self.order)
        result = self.text[index : index + self.order]

        for i in range(length):
            state = tuple(result[len(result) - self.order:])
            next_word = np.random.choice(self.graph[state])
            result.append(next_word)
        
        return " ".join(result[self.order:])


markov = Markov(3)
markov.fit("data/Bible.txt")
markov.generate(100)



