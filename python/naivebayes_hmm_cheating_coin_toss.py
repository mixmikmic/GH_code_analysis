from pomegranate import *
import numpy as np
get_ipython().run_line_magic('pylab', 'inline')

rigged = State( DiscreteDistribution({'H': 0.8, 'T': 0.2}), name="rigged" )
unrigged = State( DiscreteDistribution({'H': 0.5, 'T': 0.5}), name="unrigged" )

non_cheater = HiddenMarkovModel( name="non-cheater" )
non_cheater.add_state( unrigged )

dumb_cheater = HiddenMarkovModel( name="dumb-cheater" )
dumb_cheater.add_state( rigged )

non_cheater.start = unrigged
dumb_cheater.start = rigged

non_cheater.add_transition( unrigged, unrigged, 1 )
dumb_cheater.add_transition( rigged, rigged, 1 )

non_cheater.bake()
dumb_cheater.bake()

smart_cheater = HiddenMarkovModel( name="smart-cheater" )

smart_cheater.add_transition( smart_cheater.start, unrigged, 0.5 )
smart_cheater.add_transition( smart_cheater.start, rigged, 0.5 )

smart_cheater.add_transition( rigged, rigged, 0.5 )
smart_cheater.add_transition( rigged, unrigged, 0.5 )
smart_cheater.add_transition( unrigged, rigged, 0.5 )
smart_cheater.add_transition( unrigged, unrigged, 0.5 )

smart_cheater.bake()

plt.title("smart cheater hmm")
smart_cheater.plot()

plt.title("dumb cheater hmm")
dumb_cheater.plot()

plt.title("non-cheater hmm")
non_cheater.plot()

players = NaiveBayes([ non_cheater, smart_cheater, dumb_cheater ])

data = np.array([list( 'HHHHHTHTHTTTTHHHTHHTTHHHHHTH' ),
                 list( 'HHHHHHHTHHHHTTHHHHHHHTTHHHHH' ),
                 list( 'THTHTHTHTHTHTTHHTHHHHTTHHHTT' )])

probs = players.predict_proba( data )

for i in range(len(probs)):
    print "For sequence {}, {:.3}% non-cheater, {:.3}% smart cheater, {:.3}% dumb cheater.".format( i+1, 100*probs[i][0], 100*probs[i][1], 100*probs[i][2])

output = players.predict( data )

for i in range(len(output)):
    print "Sequence {} is a {}".format( i+1, "non-cheater" if output[i] == 0 else "smart cheater" if output[i] == 1 else "dumb cheater")

X = np.array([list( 'HHHHHTHTHTTTTH' ),
              list( 'HHTHHTTHHHHHTH' )])

y = np.array([ 1, 1 ])

players.fit( X, y )

