import dallinger

experiment = dallinger.experiments.Bartlett1932()
data = experiment.collect("3b9c2aeb-0eb7-4432-803e-bc437e17b3bb")
data.infos.df

