from pomegranate import *
import seaborn
seaborn.set_style('whitegrid')
get_ipython().run_line_magic('pylab', 'inline')

male = NormalDistribution.from_samples([ 6.0, 5.92, 5.58, 5.92, 6.08, 5.83 ])
female = NormalDistribution.from_samples([ 5.0, 5.5, 5.42, 5.75, 5.17, 5.0 ])

male.plot( n=100000, edgecolor='c', color='c', bins=50, label='Male' )
female.plot( n=100000, edgecolor='g', color='g', bins=50, label='Female' )
plt.legend( fontsize=14 )
plt.ylabel('Count')
plt.xlabel('Height (ft)')
plt.show()

print "Male distribution has mu = {:.3} and sigma = {:.3}".format( *male.parameters )
print "Female distribution has mu = {:.3} and sigma = {:.3}".format( *female.parameters )

clf = NaiveBayes([ male, female ])

data = np.array([[5.0], [6.0], [4.92], [5.5]])

for sample, probability in zip( data, clf.predict_proba(data) ):
    print "Height {:5.5}, {:5.5}% chance male and {:5.5}% chance female".format( sample, 100*probability[0], 100*probability[1])

for sample, result in zip( data, clf.predict( data )):
    print "Person with height {} is {}.".format( sample, "female" if result else "male" )

X = np.array([[180], [190], [170], [165], [100], [150], [130], [150]])
y = np.array([ 0, 0, 0, 0, 1, 1, 1, 1 ])

clf.fit( X, y )

data = np.array([[130], [200], [100], [162], [145]])

for sample, result in zip( data, clf.predict( data )):
    print "Person with weight {} is {}.".format( sample, "female" if result else "male" )

