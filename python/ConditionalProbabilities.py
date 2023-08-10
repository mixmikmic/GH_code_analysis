get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np 

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
            
        

# samples to generate 
num_samples = 1000

## Prior probabilities of a song being jazz or country 
values = ['country', 'jazz']
probs = [0.7, 0.3]
genre = Random_Variable('genre',values, probs)

# conditional probabilities of a song having lyrics or not given the genre 
values = ['no', 'yes']
probs = [0.9, 0.1] 
lyrics_if_jazz = Random_Variable('lyrics_if_jazz', values, probs)

values = ['no', 'yes']
probs = [0.2, 0.8]
lyrics_if_country = Random_Variable('lyrics_if_country', values, probs)

# conditional generating proces first sample prior and then based on outcome 
# choose which conditional probability distribution to use 

random_lyrics_samples = [] 
for n in range(num_samples): 
    # the 1 below is to get one sample and the 0 to get the first item of the list of samples 
    random_genre_sample = genre.sample(1)[0]

    # depending on the outcome of the genre sampling sample the appropriate 
    # conditional probability 
    if (random_genre_sample == 'jazz'): 
        random_lyrics_sample = (lyrics_if_jazz.sample(1)[0], 'jazz')
    else: 
        random_lyrics_sample = (lyrics_if_country.sample(1)[0], 'country')
    random_lyrics_samples.append(random_lyrics_sample)

# output 1 item per line and output the first 20 samples 
for s in random_lyrics_samples[0:20]: 
    print(s) 
        

# First only consider jazz samples 
jazz_samples = [x for x in random_lyrics_samples if x[1] == 'jazz']
for s in jazz_samples[0:20]: 
    print(s) 


est_no_if_jazz = len([x for x in jazz_samples if x[0] == 'no']) / len(jazz_samples)
est_yes_if_jazz = len([x for x in jazz_samples if x[0] == 'yes']) / len(jazz_samples)
print(est_no_if_jazz, est_yes_if_jazz)

no_samples = [x for x in random_lyrics_samples if x[0] == 'no']
est_jazz_if_no_lyrics = len([x for x in no_samples if x[1] == 'jazz']) / len(no_samples)
print(est_jazz_if_no_lyrics)



