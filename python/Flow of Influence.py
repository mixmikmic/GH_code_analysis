from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, State

# helper function to pretty print observation results
def update(network, observations, variables=None):
    beliefs = network.forward_backward(observations)
    for state, dist in zip(network.states, beliefs):
        if variables is None or state.name in variables:
            fixed = {}
            for k, v in dist.parameters[0].items():
                fixed[k] = "{:.2}".format(v)
            print("{:<15}\t{}".format(state.name, fixed))

difficulty = DiscreteDistribution({'hard': 0.6, 'easy': 0.4})
intelligence = DiscreteDistribution({'high': 0.7, 'low': 0.3})
grade = ConditionalProbabilityTable([
        ['hard', 'high', 'A', 0.3],
        ['hard', 'high', 'B', 0.4],
        ['hard', 'high', 'C', 0.3],
        ['hard', 'low', 'A', 0.05],
        ['hard', 'low', 'B', 0.25],
        ['hard', 'low', 'C', 0.7],
        ['easy', 'high', 'A', 0.9],
        ['easy', 'high', 'B', 0.08],
        ['easy', 'high', 'C', 0.02],
        ['easy', 'low', 'A', 0.5],
        ['easy', 'low', 'B', 0.3],
        ['easy', 'low', 'C', 0.2]
    ], [difficulty, intelligence])
sat = ConditionalProbabilityTable([
        ['high', 'good', 0.95],
        ['high', 'bad', 0.05],
        ['low', 'good', 0.2],
        ['low', 'bad', 0.8],
    ], [intelligence])
letter = ConditionalProbabilityTable([
        ['A', 'unfavorable', 0.1],
        ['A', 'favorable', 0.9],
        ['B', 'unfavorable', 0.4],
        ['B', 'favorable', 0.6],
        ['C', 'unfavorable', 0.99],
        ['C', 'favorable', 0.01],
    ], [grade])

difficulty_state = State(difficulty, name='difficulty')
intelligence_state = State(intelligence, name='intelligence')
grade_state = State(grade, name='grade')
sat_state = State(sat, name='sat')
letter_state = State(letter, name='letter')

student = BayesianNetwork("Student Network")
student.add_states([difficulty_state, intelligence_state, grade_state, sat_state, letter_state])
student.add_transition(difficulty_state, grade_state)
student.add_transition(intelligence_state, grade_state)
student.add_transition(intelligence_state, sat_state)
student.add_transition(grade_state, letter_state)
student.bake()

update(student, {'intelligence': 'high'}, 'sat')
update(student, {'intelligence': 'low'}, 'sat')

update(student, {'sat': 'good'}, 'intelligence')
update(student, {'sat': 'bad'}, 'intelligence')

update(student, {'difficulty': 'hard'}, 'letter')
update(student, {'difficulty': 'easy'}, 'letter')

update(student, {'letter': 'unfavorable'}, 'difficulty')
update(student, {'letter': 'favorable'}, 'difficulty')

update(student, {'letter': 'unfavorable'}, 'sat')
update(student, {'letter': 'favorable'}, 'sat')

update(student, {'sat': 'good'}, 'letter')
update(student, {'sat': 'bad'}, 'letter')

update(student, {'difficulty': 'hard'}, 'intelligence')
update(student, {'difficulty': 'easy'}, 'intelligence')

update(student, {'difficulty': 'hard', 'grade': 'A'}, 'intelligence')
update(student, {'difficulty': 'easy', 'grade': 'A'}, 'intelligence')



