get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np 

class Random_Variable: 
    
    def __init__(self, name, values, probability_distribution): 
        self.name = name 
        self.values = values 
        self.probability_distribution = probability_distribution 
        if all(type(item) is np.int64 for item in self.values): 
            self.type = 'numeric'
            self.rv = stats.rv_discrete(name = name, 
                        values = (values, probability_distribution))
        elif all(type(item) is str for item in values): 
            self.type = 'symbolic'
            self.rv = stats.rv_discrete(name = name, 
                        values = (np.arange(len(values)), probability_distribution))
            self.symbolic_values = values 
        else: 
            self.type = 'undefined'
            
    def sample(self,size): 
        if (self.type =='numeric'): 
            return self.rv.rvs(size=size)
        elif (self.type == 'symbolic'): 
            numeric_samples = self.rv.rvs(size=size)
            mapped_samples = [self.values[x] for x in numeric_samples]
            return mapped_samples 
            
        

values = ['H', 'T']
probabilities = [0.5, 0.5]
coin = Random_Variable('coin', values, probabilities)
samples = coin.sample(50)
print(samples)

values = ['1', '2', '3', '4', '5', '6']
probabilities = [1/6.] * 6
dice = Random_Variable('dice', values, probabilities)
samples = dice.sample(10)
print(samples)

values = np.arange(1,7)
probabilities = [1/6.] * 6
dice = Random_Variable('dice', values, probabilities)
samples = dice.sample(200)
plt.stem(samples, markerfmt= ' ')

plt.figure()
plt.hist(samples,bins=[1,2,3,4,5,6,7],normed=1, rwidth=0.5,align='left');

plt.hist(samples,bins=[1,2,3,4,5,6,7],normed=1, rwidth=0.5,align='left', cumulative=True);


# we can also write the predicates directly using lambda notation 
est_even = len([x for x in samples if x%2==0]) / len(samples)
est_2 = len([x for x in samples if x==2]) / len(samples)
est_4 = len([x for x in samples if x==4]) / len(samples)
est_6 = len([x for x in samples if x==6]) / len(samples)
print(est_even)
# Let's print some estimates 
print('Estimates of 2,4,6 = ', (est_2, est_4, est_6))
print('Direct estimate = ', est_even) 
print('Sum of estimates = ', est_2 + est_4 + est_6)
print('Theoretical value = ', 0.5)

import music21 as m21 
from music21 import midi 

littleMelody = m21.converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#")
sp = midi.realtime.StreamPlayer(littleMelody)
littleMelody.show()
sp.play()

values = ['c4','c#4','d4','d#4', 'e4', 'f4','f#4', 'g4','g#4','a4','a#4','b4']
probabilities = [1/12.]*12 
note_rv = Random_Variable('note', values, probabilities)
print(note_rv.sample(10))
note_string = 'tinynotation: 4/4 ; ' + " ".join(note_rv.sample(10))
print(note_string)
randomMelody = m21.converter.parse(note_string)
sp = midi.realtime.StreamPlayer(randomMelody)
randomMelody.show()
sp.play()

bachPiece = m21.corpus.parse('bwv66.6')
sp = midi.realtime.StreamPlayer(bachPiece)
bachPiece.show()
sp.play()

chorales = m21.corpus.search('bach', fileExtensions='xml')
print(chorales)

chorale_probs = np.zeros(12)
totalNotes = 0 
transition_matrix = np.ones([12,12])     # hack of adding one to avoid 0 counts 
j = 0

for (i,chorale) in enumerate(chorales): 
    score = chorale.parse()
    analyzedKey = score.analyze('key')
    # only consider C major chorales 
    if (analyzedKey.mode == 'major') and (analyzedKey.tonic.name == 'C'): 
        j = j + 1 
        print(j,chorale)
        score.parts[0].pitches
        for (i,p) in enumerate(score.parts[0].pitches):  
            chorale_probs[p.pitchClass] += 1
            if i < len(score.parts[0].pitches)-1:
                transition_matrix[p.pitchClass][score.parts[0].pitches[i+1].pitchClass] += 1 
            totalNotes += 1

chorale_probs /= totalNotes  
transition_matrix /= transition_matrix.sum(axis=1,keepdims=1)
print('Finished processing')

print("Note probabilities:")
print(chorale_probs)
print("Transition matrix:")
print(transition_matrix)


values = ['c4','c#4','d4','d#4', 'e4', 'f4','f#4', 'g4','g#4','a4','a#4','b4']
note_rv = Random_Variable('note', values, chorale_probs)
note_string = 'tinynotation: 4/4 ; ' + " ".join(note_rv.sample(16))
print(note_string)
randomChoraleMelody = m21.converter.parse(note_string)
sp = midi.realtime.StreamPlayer(randomChoraleMelody)
randomChoraleMelody.show()
sp.play()

def markov_chain(transmat, state, state_names, samples): 
    (rows, cols) = transmat.shape 
    rvs = [] 
    values = list(np.arange(0,rows))
    
    # create random variables for each row of transition matrix 
    for r in range(rows): 
        rv = Random_Variable("row" + str(r), values, transmat[r])
        rvs.append(rv)
    
    # start from initial state given as an argument and then sample the appropriate 
    # random variable based on the state following the transitions 
    states = [] 
    for n in range(samples): 
        state = rvs[state].sample(1)[0]    
        states.append(state_names[state])
    return states

note_string = 'tinynotation: 4/4 ; ' + " ".join(markov_chain(transition_matrix,0,values, 16))
print(note_string)
markovChoraleMelody = m21.converter.parse(note_string)
markovChoraleMelody.show()
sp = midi.realtime.StreamPlayer(markovChoraleMelody)
sp.play()



