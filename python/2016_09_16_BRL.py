import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gammaln
from scipy.stats import poisson, beta
from collections import defaultdict, Counter, namedtuple
from fim import fpgrowth #this is PyFIM, available from http://www.borgelt.net/pyfim.html
from bitarray import bitarray

get_ipython().magic('matplotlib inline')

class RuleListFitter(object):
    """
    Bayesian Rule List
    
    References:
    https://arxiv.org/abs/1511.01644
    https://arxiv.org/abs/1602.08610
    """
    
    def __init__(self, lambda_=2, eta=2, niter=10000, chains=3, warmup=None, thinning=1,
                 minsupport=10, maxlhs=2, alpha=(1.,1.), verbose=False):
        self.lambda_ = lambda_
        self.eta = eta
        self.niter = niter
        self.chains = chains
        if warmup >= 0:
            self.warmup = warmup
        else:
            self.warmup = self.niter//2
        self.thinning = thinning
        self.minsupport = minsupport
        self.maxlhs = maxlhs
        self.alpha = alpha
        self.verbose = verbose
        self.rules = None
        self.jinit = None
        self.estimate = None
        self.RuleMetadata = namedtuple("RuleMetadata",
                                       "rulelist N log_posterior n_rules cardinality_sum theta theta_low theta_high")
    
    def __repr__(self):
        return """<RuleListFitter lambda_=%r, eta=%r, niter=%r, chains=%r, warmup=%r, thinning=%r, 
minsupport=%r, maxlhs=%r, alpha=%r, verbose=%r>
               """ % (self.lambda_,self.eta,self.niter,self.chains, self.warmup, self.thinning,
                      self.minsupport, self.maxlhs, self.alpha, self.verbose)
    
    def __str__(self):
        return self.tostring()
    
    def tostring(self):
        str_ = ''
        if self.estimate:
            for i, rule in enumerate(self.estimate.rulelist):
                if i==0:
                    str_ += 'IF (' + " AND ".join([str(r) for r in rule]) + ") THEN "
                elif i==len(self.estimate.rulelist)-1:
                    str_ += 'ELSE '
                else:
                    str_ += 'ELSE IF (' + " AND ".join([str(r) for r in rule]) + ") THEN "
                str_ += "survival probability %0.2f (%0.2f-%0.2f)" % (self.estimate.theta[i],
                                                                       self.estimate.theta_low[i],
                                                                       self.estimate.theta_high[i])
                str_ += '\n'
            str_ += "(log posterior: %0.3f)" % self.estimate.log_posterior
        else:
            str_ = "<Untrained RuleList>"
        return str_

    def rule_metadata(self, rulelist):
        n_rules = len(rulelist)-1
        cardinality_sum = np.sum([len(rule) for rule in rulelist if rule != ()])
        N = np.array(self.captured(rulelist))
        log_posterior = self.log_posterior(rulelist,N)
        theta = (N[:,0] + self.alpha[0])/np.array([np.sum(n + self.alpha) for n in N])
        theta_low, theta_high = beta.interval(0.95,N[:,0] + self.alpha[0],N[:,1] + self.alpha[1])
        return self.RuleMetadata(rulelist, N, log_posterior, n_rules, cardinality_sum, theta, theta_low, theta_high)
     
    def fit(self, X, y):

        def mine_rules(data, minsupport, maxlhs):
            # TODO::Make column-values unique
            fp_rules = fpgrowth(
                data,
                supp=minsupport,
                zmax=maxlhs
            )
            return set([x[0] for x in fp_rules if np.nan not in x[0]])

        data_neg = X[y == 0] #negative rows
        data_pos = X[y == 1] #predictor rows
        self.positive_rows = bitarray((y==1).tolist())
        self.negative_rows = ~self.positive_rows
        
        self.rules = mine_rules(data_pos, self.minsupport, self.maxlhs)                    | mine_rules(data_neg, self.minsupport, self.maxlhs)                    | {()} #null rule
            
        if self.verbose:
            print "Number of rows:", len(X)
            print "Mined association rules:", len(self.rules)
        
        self.jinit = { rule:self.calculate_bitarray(X, rule) for rule in self.rules }
        
        self.pmf_lambda = { i:poisson.logpmf(i, self.lambda_) for i in range(len(self.rules)+2) }
        self.pmf_eta = { i:poisson.logpmf(i, self.eta) for i in range(1,self.maxlhs+1) }
        self.normalization_eta = poisson.cdf(self.maxlhs,self.eta) - poisson.pmf(0,self.eta)

        self.trace = []
        self.log_posteriors = {}
        
        for chain in xrange(self.chains): #run chains serially
            if self.verbose:
                print "Chain:", chain
            chain_trace = []
            chain_log_posteriors = {}
        
            rulelist = self.initialize_rulelist() #generate a seed rulelist
            chain_log_posteriors[rulelist.__str__()] = [self.rule_metadata(rulelist), 0]

            for i in xrange(self.niter):
                rulelist_star, logQ = self.mutate_rulelist(rulelist) # generate proposal

                if rulelist_star.__str__() not in chain_log_posteriors:
                    chain_log_posteriors[rulelist_star.__str__()] = [self.rule_metadata(rulelist_star),0]

                # M-H coefficient
                r = min(1, np.exp(  chain_log_posteriors[rulelist_star.__str__()][0].log_posterior
                                  - chain_log_posteriors[rulelist.__str__()][0].log_posterior
                                  + logQ))

                if np.random.uniform() < r:
                    rulelist = rulelist_star #accept proposal

                if (i >= self.warmup) and (i % self.thinning == 0): #warmup and thinning (if any)
                    chain_log_posteriors[rulelist.__str__()][1] += 1
                    chain_trace.append(rulelist)
                    
            #merge individual chain into global trace/metadata
            self.trace += chain_trace
            for key, value in chain_log_posteriors.items():
                if key not in self.log_posteriors:
                    self.log_posteriors[key] = value
                else:
                    self.log_posteriors[key][1] += value[1]
                    
        self.estimate = self.point_estimate()
        
        return self
   
    def calculate_bitarray(self, X, rule):
        rule_set = set(rule)
        return bitarray([rule_set <= set(x) for x in X])

    def initialize_rulelist(self): #self.rules, self.lambda_, self.eta, self.verbose
        # Sample a decision list length m \sim p(m|λ)
        m = np.Inf
        while m >= len(self.rules): # Initial list can be zero as long as we add into it later
            m = poisson.rvs(self.lambda_)
            
        avaliable_rules = self.rules.copy()
        d = []
        for _ in range(m):
            #Sample the cardinality of antecedent aj in d as cj \sim p(cj|c<j,A,η).
            avaliable_cardinalities = Counter([len(r) for r in avaliable_rules if len(r)>0]).keys()
            c = 0
            while c==0 or c>max(avaliable_cardinalities) or c not in avaliable_cardinalities:
                c = poisson.rvs(self.eta)        

            #Sample aj of cardinality cj from p(aj|a<j,cj,A).
            rules = [r for r in avaliable_rules if len(r) == c]
            rule = rules[np.random.randint(len(rules))]
            
            avaliable_rules = avaliable_rules - set([rule])
            d.append(rule)
            
        d.append(()) #null rule
        
        if self.verbose:
            print "Initial rulelist (m=%d):" % m,
            print d

        return d

    
    def captured(self, rulelist): #self.jinit
        jcaptured = {}
        
        captured = ~self.jinit[()]
        N = []
        
        for rule in rulelist:
            #j.captures ← j.init ∨ ¬captured
            #captured ← j.captures ∧ captured
            jcaptured[rule] = self.jinit[rule] & (~captured)
            captured = jcaptured[rule] ^ captured
            N.append([(jcaptured[rule] & self.positive_rows).count(),(jcaptured[rule] & self.negative_rows).count()])
        
        return N

            
    def log_posterior(self, rulelist, N):
        log_likelihood = self.log_likelihood(N)
        log_prior = self.log_prior(rulelist)
        return log_likelihood + log_prior
    
    
    def log_likelihood(self, N):
        """
        p(y|x, d, α) =
        \prod_{j=1}^{m} \frac{\gamma(N_{j,+}+α_0)\gamma(N_{j,-}+α_1)}
                             {\sum(\gamma(\gamma(N_{j,+}+\gamma(N_{j,-}+α_0+α_1))}
        """
        numerator = gammaln(N+self.alpha)
        denomerator = gammaln(np.sum(N+self.alpha,axis=1))
        return np.sum(numerator) - np.sum(denomerator)
    
    
    def log_prior(self, rulelist): #self.rules, self.pmf_lambda, self.pmf_eta, self.normalization_eta
        """
        p(d|A,λ,η) = p(m|A,λ) \prod_{j=1}^{m} p(cj|c<j,A,η) p(aj|a<j,cj,A)
        """
                
        all_rules = set([r for r in self.rules if len(r)>0])
        all_cardinalities = set(Counter([len(r) for r in all_rules]).keys())
        
        rulelist = [rule for rule in rulelist if len(rule)>0]
        log_prior = self.pmf_lambda[len(rulelist)]

        remaining_rules = all_rules.copy()
        for rule in rulelist:
            
            cardinalities = Counter([len(r) for r in remaining_rules])            
            remaining_cardinalities = set(cardinalities.keys())
            eliminated_cardinalities = all_cardinalities - remaining_cardinalities

            log_prior += self.pmf_eta[len(rule)]
            log_prior -= np.log(self.normalization_eta - sum([self.pmf_eta[l] for l in eliminated_cardinalities]))
            log_prior -= np.log(cardinalities[len(rule)])
            
            remaining_rules = remaining_rules - set([rule])

        return log_prior
    

    def mutate_rulelist(self, rulelist): #self.rules

        rulelist = [rule for rule in rulelist if len(rule)>0]
        
        # calculate the PMF distribution for the available mutation paths
        if len(rulelist) == 0: # No rules in the rulelist yet -- can only insert new rule
            path_probabilites = [0, 1., 0]
        elif len(rulelist) == 1: # Only one rule -- can only insert and remove
            path_probabilites = [0, 0.5, 0.5]
        elif len(rulelist) == len(self.rules) - 1: # List have every possible rule, can only swap or remove
            path_probabilites = [0.5, 0, 0.5]
        elif len(rulelist) == len(self.rules) - 2: # Only one rule remaining, so the inverse probabilites have a correction
            path_probabilites = [1./3, 1./3, 1./3]
        else: # All paths possible
            path_probabilites = [1./3, 1./3, 1./3]
            
        mutation = np.random.choice(['swap','insert','remove'], p=path_probabilites)

        # Q(d|d*)    p(d|d*,swap*)p(swap*) + p(d|d*,insert*)p(insert*) + p(d|d*,remove*)p(remove*)
        # ------- =  -----------------------------------------------------------------------------
        # Q(d*|d)       p(d*|d,swap)p(swap) + p(d*|d,insert)p(insert) + p(d*|d,remove)p(remove)
        
        if mutation == 'swap':
            Q = 1.0
        elif mutation == 'insert':
            if len(rulelist) == 0:
                # After an insert, we can only get back to an empty list via a remove
                # But there are two possible operations from the d* state (insert, or remove), so p(remove*)=0.5
                
                # Q(d|d*)     0 + 0*0.5 + (1/|d*|)*0.5           
                # ------- =   --------------------       =  0.5*(float(len(self.rules)-1-len(rulelist)))
                # Q(d*|d)     0 + 0 + 1/((|a|-|d|)(|d|+1))*1.0 
                Q = (0.5)*(float(len(self.rules)-1-len(rulelist)))
            elif len(rulelist) == 1:
                # Q(d|d*)    0*(1/3) + 0*(1/3) + (1/|d*|)*(1/3)
                # ------- =  ----------------------------------        = (2/3)*(float(len(self.rules)-1-len(rulelist)))
                # Q(d*|d)    0 + 1/((|a|-|d|)(|d|+1))*(1/2) + 0*(1/2)
                Q = (2./3)*(float(len(self.rules)-1-len(rulelist)))
            elif len(rulelist) == len(self.rules) - 2:
                # Q(d|d*)    0*(1/2) + 0*0 + (1/|d*|)*(1/2)
                # ------- =  ----------------------------------        = (3/2)*(float(len(self.rules)-1-len(rulelist)))
                # Q(d*|d)    0*(1/3) + 1/((|a|-|d|)(|d|+1))*(1/3) + 0*(1/3)
                Q = (3./2)*(float(len(self.rules)-1-len(rulelist)))
            else:
                # Q(d|d*)    0*(1/3) + 0*(1/3) + (1/|d*|)*(1/3)
                # ------- =  ----------------------------------        = (1)*(float(len(self.rules)-1-len(rulelist)))
                # Q(d*|d)    0*(1/3) + 1/((|a|-|d|)(|d|+1))*(1/3) + 0*(1/3)
                Q = (1.)*(float(len(self.rules)-1-len(rulelist)))
        elif mutation == 'remove':
            if len(rulelist) == 1:
                # Q(d|d*)    0*(0) + 1/((|a|-|d*|)(|d*|+1))*(1) + 0*(0)      
                # ------- =  ----------------------------------        = (2)/(float(len(self.rules)-len(rulelist)))
                # Q(d*|d)    0 + 0*(1/2) + 1/|d|*(1/2)
                Q = (2)/(float(len(self.rules)-1.-len(rulelist)-1.))
            elif len(rulelist) == len(self.rules) - 1:
                # Q(d|d*)    0*(1/3) + 1/((|a|-|d*|)(|d*|+1))*(1/3) + 0*(1./3)
                # ------- =  ----------------------------------        = (2./3)/(float(len(self.rules)-len(rulelist)))
                # Q(d*|d)    0*(1/2) + 0*(0) + 1/|d|*(1/2)
                Q = (2./3)/(float(len(self.rules)-1.-len(rulelist)-1.))                
            elif len(rulelist) == len(self.rules) - 2:
                # Q(d|d*)    0*(1/3) + 1/((|a|-|d*|)(|d*|+1))*(1/3) + 0*(1./3)
                # ------- =  ----------------------------------        = (1.)/(float(len(self.rules)-len(rulelist)))
                # Q(d*|d)    0*(1/3) + 0*(1/3) + 1/|d|*(1/3)
                Q = (1.)/(float(len(self.rules)-1.-len(rulelist)-1.))                
            else:
                # Q(d|d*)    0*(1/3) + 1/((|a|-|d*|)(|d*|+1))*(1/3) + 0*(1./3)
                # ------- =  ----------------------------------        = (1.)/(float(len(self.rules)-len(rulelist)))
                # Q(d*|d)    0*(1/3) + 0*(1/3) + 1/|d|*(1/3)
                Q = (1.)/(float(len(self.rules)-1.-len(rulelist)-1.))                
        else:
            raise

        # perform the mutation
        if mutation == 'swap':
            a,b = np.random.permutation(range(len(rulelist)))[:2]
            rulelist[a], rulelist[b] = rulelist[b], rulelist[a]
        elif mutation == 'insert':
            try:
                new_rules = list(set(self.rules) - set(rulelist) - set([()]))
                new_rule = new_rules[np.random.randint(len(new_rules))]
            except:
                print rulelist
                print list(set(self.rules) - set(rulelist) - set([()]))
            rulelist.insert(np.random.randint(len(rulelist)+1), new_rule)
        elif mutation == 'remove':
            rulelist.pop(np.random.randint(len(rulelist)))
        else:
            raise
            
        rulelist.append(())
        return rulelist, np.log(Q)
    
    def point_estimate(self): #self.log_posteriors, self.verbose
        if len(self.log_posteriors) == 0:
            return None
                
        #find the average rule length and width
        lengths = 0.
        widths = 0.
        n = 0
        n_rules = 0
        for rulelist in self.log_posteriors.values():
            lengths += rulelist[0].n_rules * rulelist[1]
            widths += rulelist[0].cardinality_sum
            n += rulelist[1]
            n_rules += rulelist[0].n_rules
            
        avg_length = lengths/n
        avg_width = widths/n_rules
            
        if self.verbose:
            print "Posterior average length:", avg_length
            print "Posterior average width:", avg_width
            
        #filter for only rulelists around the average
        min_length = int(np.floor(avg_length))
        min_width  = int(np.floor(avg_width))
        max_length = int(np.ceil(avg_length))
        max_width  = int(np.ceil(avg_width))
        
        keys = []
        posteriors = []
        for key,rulelist in self.log_posteriors.items():
            metadata = rulelist[0]
            try:
                avg_cardinality = float(metadata.cardinality_sum)/float(metadata.n_rules)
            except:
                print metadata
                continue
            if metadata.n_rules >= min_length and metadata.n_rules <= max_length                     and avg_cardinality >= min_width and avg_cardinality <= max_width:
                keys.append(key)
                posteriors.append(metadata.log_posterior)
                
        #return rulelist with maximum posterior value
        max_key = keys[np.argmax(posteriors)]
        return self.log_posteriors[max_key][0]
    
    def predict_point_estimate(self, X, rulelist_metadata):
        jinit = { rule:self.calculate_bitarray(X, rule) for rule in self.rules }
        
        captured = ~jinit[()]
        predictions = -1.*np.zeros(len(X))
        
        for i, rule in enumerate(rulelist_metadata.rulelist):
            #j.captures ← j.init ∨ ¬captured
            #captured ← j.captures ∧ captured
            jcaptured = jinit[rule] & (~captured)
            captured = jcaptured ^ captured

            predictions[np.array(jcaptured.tolist())] = rulelist_metadata.theta[i]
            
        return predictions

    def predict_posterior(self, X):
        posterior_samples = 0
        predictions = np.zeros(len(X))
        
        if self.verbose:
            print "number of rules in the posterior set:", len(self.log_posteriors)
        
        for i,(k,v) in enumerate(self.log_posteriors.items()):
            if i%10000 == 0 and self.verbose:
                print "i:", i
            rulelist_metadata, n = v
            if n==0:
                continue
            posterior_samples += n
            predictions += n*self.predict_point_estimate(X, rulelist_metadata)
            
        if posterior_samples==0:
            return None
        else:
            return predictions/posterior_samples
    
    def predict_proba(self, X, mode='point'):
        if mode=='point':
            return self.predict_point_estimate(X, self.estimate)
        elif mode=='posterior':
            return self.predict_posterior(X)
        else:
            return None
    
    def predict(self, X, mode='point'):
        return (self.predict_proba(X, mode)>=0.5).astype(int)

data = pd.read_csv('train.csv')

data['binned_age'] = pd.cut(data['Age'], [0,18,np.Inf], labels=['child','adult'])
data['cabin_class'] = pd.cut(data['Pclass'], [0,1,2,3], labels=['1st class', '2nd class', '3rd class'])

X = data[['cabin_class','Sex','binned_age']].values
y = data['Survived'].values

rl = RuleListFitter(lambda_=3, eta=2, chains=4, niter=50000, maxlhs=3, verbose=True)

get_ipython().run_cell_magic('time', '', 'rl.fit(X, y)')

print rl

test_data = pd.read_csv('test.csv')

test_data['binned_age'] = pd.cut(test_data['Age'], [0,18,np.Inf], labels=['child','adult'])
test_data['cabin_class'] = pd.cut(test_data['Pclass'], [0,1,2,3], labels=['1st class', '2nd class', '3rd class'])

X_test = test_data[['cabin_class','Sex','binned_age']].values

rl.predict_proba(X_test, mode='point')[:10]

predictions = rl.predict(X_test)
predictions[:10]

predictions_out = pd.Series(predictions, index=test_data.PassengerId, name="Survived")
predictions_out.to_csv("BRL_predictions_point.csv", header=True)

get_ipython().system('head BRL_predictions.csv')

get_ipython().run_cell_magic('time', '', "posterior_predictions = rl.predict(X_test, mode='posterior')")

posterior_predictions_out = pd.Series(posterior_predictions, index=test_data.PassengerId, name="Survived")
posterior_predictions_out.to_csv("BRL_predictions_posterior.csv", header=True)

get_ipython().system('head BRL_predictions_posterior.csv')

