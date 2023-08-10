from __future__ import print_function, division

from thinkbayes2 import Pmf, Suite

from fractions import Fraction

class Test(Suite):
    """Represents beliefs about a patient based on a medical test."""
    
    def __init__(self, p, s, t, label='Test'):
        # initialize the prior probabilities
        d = dict(sick=p, notsick=1-p)
        super(Test, self).__init__(d, label)
        
        # store the parameters
        self.p = p
        self.s = s
        self.t = t
        
        # make a nested dictionary to compute likelihoods
        self.likelihood = dict(pos=dict(sick=s, notsick=t),
                               neg=dict(sick=1-s, notsick=1-t))
        
    def Likelihood(self, data, hypo):
        """
        data: 'pos' or 'neg'
        hypo: 'sick' or 'notsick'
        """
        return self.likelihood[data][hypo]

p = Fraction(1, 10)     # prevalence
s = Fraction(9, 10)     # sensitivity
t = Fraction(3, 10)     # false positive rate

test = Test(p, s, t)
test.Print()

test.likelihood

test.Update('pos')
test.Print()

class MetaTest(Suite):
    """Represents a set of tests with different values of `t`."""
    
    def Likelihood(self, data, hypo):
        """
        data: 'pos' or 'neg'
        hypo: Test object
        """
        # the return value from `Update` is the total probability of the
        # data for a hypothetical value of `t`
        return hypo.Update(data)

q = Fraction(1, 2)
t1 = Fraction(2, 10)
t2 = Fraction(4, 10)

test1 = Test(p, s, t1, 'Test(t=0.2)')
test2 = Test(p, s, t2, 'Test(t=0.4)')

metatest = MetaTest({test1:q, test2:1-q})
metatest.Print()

metatest.Update('pos')

metatest.Print()

def MakeMixture(metapmf, label='mix'):
    """Make a mixture distribution.

    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      label: string label for the new Pmf.

    Returns: Pmf object.
    """
    mix = Pmf(label=label)
    for pmf, p1 in metapmf.Items():
        for x, p2 in pmf.Items():
            mix.Incr(x, p1 * p2)
    return mix

predictive = MakeMixture(metatest)
predictive.Print()

def MakeMetaTest(p, s, pmf_t):
    """Makes a MetaTest object with the given parameters.
        
    p: prevalence
    s: sensitivity
    pmf_t: Pmf of possible values for `t`
    """
    tests = {}
    for t, q in pmf_t.Items():
        label = 'Test(t=%s)' % str(t)
        tests[Test(p, s, t, label)] = q
    return MetaTest(tests)

def Marginal(metatest):
    """Extracts the marginal distribution of t.
    """
    marginal = Pmf()
    for test, prob in metatest.Items():
        marginal[test.t] = prob
    return marginal

def Conditional(metatest, t):
    """Extracts the distribution of sick/notsick conditioned on t."""
    for test, prob in metatest.Items():
        if test.t == t:
            return test

pmf_t = Pmf({t1:q, t2:1-q})
metatest = MakeMetaTest(p, s, pmf_t)
metatest.Print()

metatest = MakeMetaTest(p, s, pmf_t)
metatest.Update('pos')
metatest.Print()

Marginal(metatest).Print()

cond1 = Conditional(metatest, t1)
cond1.Print()

cond2 = Conditional(metatest, t2)
cond2.Print()

MakeMixture(metatest).Print()

convolution = metatest + metatest
convolution.Print()

marginal = MakeMixture(metatest+metatest)
marginal.Print()

marginal = MakeMixture(metatest) + MakeMixture(metatest)
marginal.Print()

from random import random

def flip(p):
    return random() < p

def generate_pair_A(p, s, pmf_t):
    while True:
        sick1, sick2 = flip(p), flip(p)
        
        t = pmf_t.Random()
        test1 = flip(s) if sick1 else flip(t)

        t = pmf_t.Random()
        test2 = flip(s) if sick2 else flip(t)

        yield test1, test2, sick1, sick2

def run_simulation(generator, iters=100000):
    pmf_t = Pmf([0.2, 0.4])
    pair_iterator = generator(0.1, 0.9, pmf_t)

    outcomes = Pmf()
    for i in range(iters):
        test1, test2, sick1, sick2 = next(pair_iterator)
        if test1 and test2:
            outcomes[sick1, sick2] += 1

    outcomes.Normalize()
    return outcomes

outcomes = run_simulation(generate_pair_A)
outcomes.Print()

