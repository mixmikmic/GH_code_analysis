import thinkbayes2

prior = thinkbayes2.Suite({'State 1': 0.4, 'State 2': 0.25, 'State 3': 0.35})
prior.Print()

likelihood = {'State 1': 0.5, 'State 2': 0.60, 'State 3': 0.35}

posterior = prior.Copy()
for hypo in posterior:
    posterior[hypo] *= likelihood[hypo]
posterior.Print()

posterior.Normalize()

posterior.Print()



