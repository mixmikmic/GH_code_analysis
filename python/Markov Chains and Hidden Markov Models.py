get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from hmmlearn import hmm



class Random_Variable: 
    
    def __init__(self, name, values, probability_distribution): 
        self.name = name 
        self.values = values 
        self.probability_distribution = probability_distribution 
        if all(type(item) is np.int64 for item in values): 
            self.type = 'numeric'
            self.rv = stats.rv_discrete(name = name, values = (values, probability_distribution))
        elif all(type(item) is str for item in values): 
            self.type = 'symbolic'
            self.rv = stats.rv_discrete(name = name, values = (np.arange(len(values)), probability_distribution))
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
        
    def probs(self): 
        return self.probability_distribution
    
    def vals(self): 
        print(self.type)
        return self.values 
            
        

values = ['S', 'C']
probabilities = [0.5, 0.5]
weather = Random_Variable('weather', values, probabilities)
samples = weather.sample(365)
print(",".join(samples))

state2color = {} 
state2color['S'] = 'yellow'
state2color['C'] = 'grey'

def plot_weather_samples(samples, state2color, title): 
    colors = [state2color[x] for x in samples]
    x = np.arange(0, len(colors))
    y = np.ones(len(colors))
    plt.figure(figsize=(10,1))
    plt.bar(x, y, color=colors, width=1)
    plt.title(title)
    
plot_weather_samples(samples, state2color, 'iid')

def markov_chain(transmat, state, state_names, samples): 
    (rows, cols) = transmat.shape 
    rvs = [] 
    values = list(np.arange(0,rows))
    
    # create random variables for each row of transition matrix 
    for r in range(rows): 
        rv = Random_Variable("row" + str(r), values, transmat[r])
        rvs.append(rv)
    
    # start from initial state and then sample the appropriate 
    # random variable based on the state following the transitions 
    states = [] 
    for n in range(samples): 
        state = rvs[state].sample(1)[0]    
        states.append(state_names[state])
    return states


# transition matrices for the Markov Chain 
transmat1 = np.array([[0.7, 0.3], 
                    [0.2, 0.8]])

transmat2 = np.array([[0.9, 0.1], 
                    [0.1, 0.9]])

transmat3 = np.array([[0.5, 0.5], 
                     [0.5, 0.5]])

state2color = {} 
state2color['S'] = 'yellow'
state2color['C'] = 'grey'

# plot the iid model too
samples = weather.sample(365)
plot_weather_samples(samples, state2color, 'iid')

samples1 = markov_chain(transmat1,0,['S','C'], 365)
plot_weather_samples(samples1, state2color, 'markov chain 1')

samples2 = markov_chain(transmat2,0,['S','C'],365)
plot_weather_samples(samples2, state2color, 'marov_chain 2')

samples3 = markov_chain(transmat3,0,['S','C'], 365)
plot_weather_samples(samples3, state2color, 'markov_chain 3')



state2color = {} 
state2color['S'] = 'yellow'
state2color['C'] = 'grey'

# generate random samples for a year 
samples = weather.sample(365)
states = markov_chain(transmat1,0,['S','C'], 365)
plot_weather_samples(states, state2color, "markov chain 1")

# create two random variables one of the sunny state and one for the cloudy 
sunny_colors = Random_Variable('sunny_colors', ['y', 'r', 'b', 'g'], 
                              [0.6, 0.3, 0.1, 0.0])
cloudy_colors = Random_Variable('cloudy_colors', ['y', 'r', 'b', 'g'], 
                               [0.0, 0.1, 0.4, 0.5])

def emit_obs(state, sunny_colors, cloudy_colors): 
    if (state == 'S'): 
        obs = sunny_colors.sample(1)[0]
    else: 
        obs = cloudy_colors.sample(1)[0]
    return obs 

# iterate over the sequence of states and emit color based on the emission probabilities 
obs = [emit_sample(s, sunny_colors, cloudy_colors) for s in states]

obs2color = {} 
obs2color['y'] = 'yellow'
obs2color['r'] = 'red'
obs2color['b'] = 'blue'
obs2color['g'] = 'grey'
plot_weather_samples(obs, obs2color, "Observed sky color")

# let's zoom in a month 
plot_weather_samples(states[0:30], state2color, 'states for a month')
plot_weather_samples(obs[0:30], obs2color, 'observations for a month')


transmat = np.array([[0.7, 0.3], 
                    [0.2, 0.8]])

start_prob = np.array([1.0, 0.0, 0.0])

# yellow and red have high probs for sunny 
# blue and grey have high probs for cloudy 
emission_probs = np.array([[0.6, 0.3, 0.1, 0.0], 
                           [0.0, 0.1, 0.4, 0.5]])

model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = start_prob 
model.transmat_ = transmat 
model.emissionprob_ = emission_probs

# sample the model - X is the observed values 
# and Z is the "hidden" states 
X, Z = model.sample(365)

# we have to re-define state2color and obj2color as the hmm-learn 
# package just outputs numbers for the states 
state2color = {} 
state2color[0] = 'yellow'
state2color[1] = 'grey'
plot_weather_samples(Z, state2color, 'states')

samples = [item for sublist in X for item in sublist]
obj2color = {} 
obj2color[0] = 'yellow'
obj2color[1] = 'red'
obj2color[2] = 'blue'
obj2color[3] = 'grey'
plot_weather_samples(samples, obj2color, 'observations')

# generate the samples 
X, Z = model.sample(1000)
# learn a new model 
estimated_model = hmm.MultinomialHMM(n_components=2, n_iter=10000).fit(X)

print("Transition matrix")
print("Estimated model:")
print(estimated_model.transmat_)
print("Original model:")
print(model.transmat_)
print("Emission probabilities")
print("Estimated model")
print(estimated_model.emissionprob_)
print("Original model")
print(model.emissionprob_)

Z2 = estimated_model.predict(X)
state2color = {} 
state2color[0] = 'yellow'
state2color[1] = 'grey'
plot_weather_samples(Z, state2color, 'Original states')
plot_weather_samples(Z2, state2color, 'Predicted states')

# note the reversal of colors for the states as the order of components is not the same. 
# we can easily fix this by change the state2color 
state2color = {} 
state2color[1] = 'yellow'
state2color[0] = 'grey'
plot_weather_samples(Z2, state2color, 'Flipped Predicted states')



X, Z = estimated_model.sample(365)

state2color = {} 
state2color[0] = 'yellow'
state2color[1] = 'grey'
plot_weather_samples(Z, state2color, 'states generated by estimated model ')

samples = [item for sublist in X for item in sublist]
obs2color = {} 
obs2color[0] = 'yellow'
obs2color[1] = 'red'
obs2color[2] = 'blue'
obs2color[3] = 'grey'
plot_weather_samples(samples, obs2color, 'observations generated by estimated model')



# probabities of each state D(II), G(V), C(I). The transitions are semi-plausible but set by hand. 
# in a full problem they would be learned from data 
transmat = np.array([[0.4, 0.4, 0.2], 
                    [0.1, 0.1, 0.8], 
                    [0.0, 0.3, 0.7]])

start_prob = np.array([1.0, 0.0, 0.0])

# the emission probabilities are also set by hand and semi-plausible and correspond 
# to the probability that a chord is dominant, minor or major 7th. Notice for example 
# that if the chord is a C(I) (the third row then it will never be a dominant chord the 
# last 0.0 in that row 
emission_probs = np.array([[0.4, 0.0, 0.4], 
                           [0.3, 0.3, 0.3],
                           [0.2, 0.8, 0.0]]) 
                          

chord_model = hmm.MultinomialHMM(n_components=2)
chord_model.startprob_ = start_prob 
chord_model.transmat_ = transmat 
chord_model.emissionprob_ = emission_probs

X, Z = chord_model.sample(10)
state2name = {} 
state2name[0] = 'D'
state2name[1] = 'G'
state2name[2] = 'C'
chords = [state2name[state] for state in Z]
print(chords)

obj2name = {}
obj2name[0] = 'min7'
obj2name[1] = 'maj7'
obj2name[2] = '7'
observations = [obj2name[item] for sublist in X for item in sublist]
print(observations)

chords = [''.join(chord) for chord in zip(chords,observations)]
print(chords)

from music21 import *

# create some chords for II, V, I 
d7 = chord.Chord(['D4','F4', 'A4', 'C5'])
dmin7 = chord.Chord(['D4','F-4', 'A4', 'C5'])
dmaj7 = chord.Chord(['D4','F#4', 'A4', 'C#5'])

c7 = d7.transpose(-2)
cmin7 = dmin7.transpose(-2)
cmaj7 = dmaj7.transpose(-2)

g7 = d7.transpose(5)
gmin7 = dmin7.transpose(5)
gmaj7 = dmaj7.transpose(5)
print(g7.pitches)

stream1 = stream.Stream()
stream1.repeatAppend(dmin7,1)
stream1.repeatAppend(g7,1)
stream1.repeatAppend(cmaj7,1)
stream1.repeatAppend(cmaj7,1)
print(stream1)

name2chord = {} 
name2chord['C7'] = c7 
name2chord['Cmin7'] = cmin7 
name2chord['Cmaj7'] = cmaj7

name2chord['D7'] = d7 
name2chord['Dmin7'] = dmin7 
name2chord['Dmaj7'] = dmaj7

name2chord['G7'] = g7 
name2chord['Gmin7'] = gmin7 
name2chord['Gmaj7'] = gmaj7


hmm_chords = stream.Stream() 
for c in chords: 
    hmm_chords.repeatAppend(name2chord[c],1)


# let's check that we can play streams of chords 
#sp = midi.realtime.StreamPlayer(stream1)
#sp.play()

# let's now play a hidden markov model generated chord sequence
print(chords)
hmm_chords.show()
sp = midi.realtime.StreamPlayer(hmm_chords)
sp.play()

