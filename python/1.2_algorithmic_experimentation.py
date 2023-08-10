import dallinger

experiment = dallinger.experiments.Griduniverse()

num_rounds = [10, 20, 40]

for n in num_rounds:
    data = experiment.run(num_rounds=n)

