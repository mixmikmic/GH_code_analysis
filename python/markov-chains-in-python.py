import numpy as np

state_dict = {'A': np.array([['A', 'B', 'C'],
                             [.2, .4, .4]]),
              'B': np.array([['A', 'C'],
                             [.4, .6]]),
              'C': np.array([['A', 'B'],
                             [.6, .4]])}

class Markov(object):

    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.state = list(self.state_dict.keys())[0]

    def check_state(self):
        print('Current State: %s' % (self.state))

    def set_state(self, state):
        self.state = state
        print('State is now: %s' % (self.state))

    def next_state(self):
        A = self.state_dict[self.state]
        self.state = np.random.choice(a=list(A[0]), p=list(A[1]))
        print('New State: %s' % (self.state))

diagram_a = Markov(state_dict)
diagram_a.check_state()

diagram_a.set_state('B')

for x in range(10):
    diagram_a.next_state()

