import operator
import itertools
import numpy as np
import seaborn as sb

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

np.seterr(all='raise')

digits = load_digits()
digit_features, digit_labels = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(digit_features, digit_labels, stratify=digit_labels,train_size=0.75, test_size=0.25)

# The exploration of the dataset by benchmark algorithms
clf = DecisionTreeClassifier(random_state=34092)
clf.fit(X_train, y_train)
pred_DTC = clf.predict(X_test)
print('Base DecisionTreeClassifier accuracy: {}'.format(clf.score(X_test, y_test)))

clf = RandomForestClassifier(random_state=34092)
clf.fit(X_train, y_train)
pred_RFC = clf.predict(X_test)
print('Base RandomForestClassifier accuracy: {}'.format(clf.score(X_test, y_test)))

clf = GradientBoostingClassifier(random_state=34092)
clf.fit(X_train, y_train)
pred_GBC = clf.predict(X_test)
print('Base GradientBoostingClassifier accuracy: {}'.format(clf.score(X_test, y_test)))

print('')

max_marks = 100
min_marks = 0.33*max_marks
diff_global_marks = np.ones(y_test.shape[0])
diff_global_marks = diff_global_marks*max_marks

def predict_unseen(population):
    forest_predictions = []
    for ind_num, individual in enumerate(population):
        func = toolbox.compile(expr=individual)
        
        
        sample_counts = [int(func(*record)) for record in X_train]
        sample_counts = [max(min(sample_count, 10), 0) for sample_count in sample_counts]
        sample = []
        for sample_index, sample_count in enumerate(sample_counts):
            sample.extend([sample_index] * sample_count)
        sample = np.array(sample)

        if len(sample) == 0:
            return 1e-20, 1e-20

        clf = DecisionTreeClassifier(random_state=34092)
        clf.fit(X_train[sample], y_train[sample])
        predictions = clf.predict(X_test)
        forest_predictions.append(predictions)
        
    from collections import Counter
    from sklearn.metrics import accuracy_score
    
    y_pred = np.array([Counter(instance_forest_predictions).most_common(1)[0][0] for instance_forest_predictions in zip(*forest_predictions)])
    #np.sum(y_test == y_pred) / len(y_test)
    print "Accuracy->"+str(np.sum(y_test == y_pred)*100/ len(y_test))

# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped('MAIN', itertools.repeat(float, digit_features.shape[1]), bool, 'Feature')

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except (ZeroDivisionError, FloatingPointError): return 1.

pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(in1, output1, output2):
    if in1: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
for val in np.arange(-10., 11.):
    pset.addTerminal(val, float)

creator.create('FitnessMax', base.Fitness, weights=(1.0, 1.0))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)

def evaluate_individual(individual):
    global diff_global_marks
    # Transform the tree expression into a callable function
    func = toolbox.compile(expr=individual)
    
    sample_counts = [int(func(*record)) for record in X_train]
    sample_counts = [max(min(sample_count, 10), 0) for sample_count in sample_counts]
    sample = []
    for sample_index, sample_count in enumerate(sample_counts):
        sample.extend([sample_index] * sample_count)
    sample = np.array(sample)
    
    if len(sample) == 0:
        return 1e-20, 1e-20
    
    clf = DecisionTreeClassifier(random_state=34092)
    clf.fit(X_train[sample], y_train[sample])
    #score = clf.score(X_test, y_test)
    t_pred = clf.predict(X_test)
    total_marks = np.sum((t_pred==y_test)*diff_global_marks)
    accuracy = np.sum((t_pred==y_test))
    
    # Updating the diff_marks
    bool_ = 1*(t_pred==y_test)
    diff_global_marks = diff_global_marks-bool_
    diff_global_marks[diff_global_marks<min_marks] = min_marks
    
    return total_marks, accuracy

toolbox.register('evaluate', evaluate_individual)
#toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

population = toolbox.population(n=max_marks)
halloffame = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('std', np.std)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('max', np.max)

cxpb = 0.5
mutpb = 0.5
lambda_ = 100
mu = max_marks
ngen = 100
verbose = True

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])


# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit

if halloffame is not None:
    halloffame.update(population)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

record = stats.compile(population) if stats is not None else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record)
if verbose:
    print logbook.stream

# Begin the generational process
global diff_global_marks
all_marks = []
all_marks.append(diff_global_marks)

for gen in range(1, ngen + 1):
    # Vary the population
    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(offspring)

    # Select the next generation population
    population[:] = toolbox.select(offspring, mu)
    print len(population)
    predict_unseen(population)
    
    # Just updating the all_marks array which has all the changed values
    all_marks.append(diff_global_marks)
    
    # Update the statistics with the new population
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream
    #return population, logbook

str(halloffame[0])



marks = pd.DataFrame(all_marks[0])

marks.columns = [str(1)]

marks_ = pd.DataFrame()
for i in range(0,len(all_marks)):
    if(i==0):
        marks_ = pd.DataFrame(all_marks[0])
    else:
        if(type(all_marks[i])!=float):
            temp_ = pd.DataFrame(all_marks[i])
            temp_.columns = [i]
            marks_ = pd.concat([marks_,temp_],axis=1)

get_ipython().magic('matplotlib inline')

plt.plot(marks_[0])

forest_predictions = []

for ind_num, individual in enumerate(population):
    func = toolbox.compile(expr=individual)
    subsample = np.array([func(*record) for record in X_train])
    
    if X_train[subsample].shape[0] == 0:
        continue
    
    clf = DecisionTreeClassifier(random_state=34092)
    clf.fit(X_train[subsample], y_train[subsample])
    predictions = clf.predict(X_test)
    forest_predictions.append(predictions)

from collections import Counter
from sklearn.metrics import accuracy_score

y_pred = np.array(
    [Counter(instance_forest_predictions).most_common(1)[0][0] for instance_forest_predictions in zip(*forest_predictions)])
#np.sum(y_test == y_pred) / len(y_test)
np.sum(y_test == y_pred)*100/ len(y_test)

func = toolbox.compile(expr=halloffame[0])
subsample = np.array([func(*record) for record in X_train])

subsample

print halloffame[0]

chk

(0.5*100)

chk = 1;



