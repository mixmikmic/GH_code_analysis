print(bin(1))

example="this is a string"
print(example)

spacereplaced=example.replace(" ","_")
print(spacereplaced)

test="a a a b"
print(example.strip('thstring'))

truth = ["odd", "even"]  # uses some human readiable text to make reading results easier
training_data = []
for x in range(1,33):
    number = "{:05d}".format(int((bin(x)).replace("0b","")))
    label = truth[x%2 == 0]
    training_data.append((number,label))
    
    
for counter, value in enumerate(training_data):
    print("Current: ",counter+1," Value: ",value[0], " Label: ",value[1])

n_even = 0
n_odd = 0

for x in training_data:
    if(x[1]=="even"):
        n_even+=1
    else:
        n_odd+=1
p_even = float(n_even)/len(training_data)
p_odd = float(n_odd)/len(training_data)
print ("p(even) = ", p_even,  "p(odd) = ", p_odd)    

n_even = len([x[0] for x in training_data if x[1] == "even"])
n_odd = len([x[0] for x in training_data if x[1] == "odd"])
p_even = float(n_even)/len(training_data)
p_odd = float(n_odd)/len(training_data)
print ("p(even) = ", p_even,  "p(odd) = ", p_odd)

test="this is a string"
print(test.count('is'))

n_zeros = 0
n_ones = 0

for x in training_data:
    n_zeros += x[0].count("0")
    n_ones += x[0].count("1")
    
total_characters = n_zeros + n_ones
p_zero = float(n_zeros)/total_characters
p_one = float(n_ones)/total_characters

print("Total Characters: ",total_characters, " P(0): ",p_zero, " P(1)",p_one)

n_zeros = sum([x[0].count("0") for x in training_data])
n_ones = sum([x[0].count("1") for x in training_data])
total_characters = n_zeros + n_ones
p_zero = float(n_zeros)/total_characters
p_one = float(n_ones)/total_characters
print("Total Characters: ",total_characters, " P(0): ",p_zero, " P(1)",p_one)

n_zeros_even = 0
n_ones_even = 0
for x in training_data:
    if x[1] == "even":
        n_zeros_even += x[0].count("0")   
        n_ones_even += x[0].count("1")
        
p_zero_given_even = float(n_zeros_even)/(n_zeros_even + n_ones_even)
p_one_given_even = float(n_ones_even)/(n_zeros_even + n_ones_even)

print ("p(0|even) = {}   P(1|even) = {} ".format(p_zero_given_even,p_one_given_even))

n_zeros_even = sum([x[0].count("0") for x in training_data if x[1] == "even"])
n_ones_even = sum([x[0].count("1") for x in training_data if x[1] == "even"])
p_zero_given_even = float(n_zeros_even)/(n_zeros_even + n_ones_even)
p_one_given_even = float(n_ones_even)/(n_zeros_even + n_ones_even)

print ("p(0|even) = {}   P(1|even) = {} ".format(p_zero_given_even,p_one_given_even))

n_zeros_odd = 0
n_ones_odd = 0
for x in training_data:
    if x[1] == "odd":
        n_zeros_odd += x[0].count("0")   
        n_ones_odd += x[0].count("1")
        
p_zero_given_odd = float(n_zeros_odd)/(n_zeros_odd + n_ones_odd)
p_one_given_odd = 1.0 - p_zero_given_odd
print ("p(0|odd) = {}   P(1|odd) = {} ".format(p_zero_given_odd,p_one_given_odd))

n_zeros_odd = sum([x[0].count("0") for x in training_data if x[1] == "odd"])
n_ones_odd = sum([x[0].count("1") for x in training_data if x[1] == "odd"])
p_zero_given_odd = float(n_zeros_odd)/(n_zeros_odd + n_ones_odd)
p_one_given_odd = 1.0 - p_zero_given_odd

print ("p(0|odd) = {}   P(1|odd) = {} ".format(p_zero_given_odd,p_one_given_odd))

def p_odd_given(sample):
    n_zeros = sample.count("0")
    n_ones = sample.count("1")
    return p_odd * (p_zero_given_odd**n_zeros) * (p_one_given_odd**n_ones)
  
def p_even_given(sample):
    n_zeros = sample.count("0")
    n_ones = sample.count("1")
    return p_even * (p_zero_given_even**n_zeros) * (p_one_given_even**n_ones)

correct = 0
for sample, gt in training_data:
    star = ""
    if truth[int(p_odd_given(sample) < p_even_given(sample))] == gt:
        correct += 1
        star = "*"

    print("Sample: ",sample,"-->"," P(odd): %.3f"%p_odd_given(sample), "P(even): %.3f"%p_even_given(sample),
         " GT: ",gt,star)

print ("="*60)
baseline_correct1 = 0

for x in training_data:
    if x[1] == "odd":
        baseline_correct1+=1

print ("Baseline (guess all \"odd\")")
print("Total Samples: ",len(training_data)," Total Correct: ",baseline_correct1, "Accuracy: %.1f"%
     (float(baseline_correct1/len(training_data))*100),"%")
print ("="*60)
print ("Naive Bayes Classifier")
print("Total Samples: ",len(training_data)," Total Correct: ",correct, "Accuracy: %.1f"%
     (float(correct/len(training_data))*100),"%")
print ("="*60)

print ("="*60)
baseline_correct = sum([1 for x in training_data if x[1] == "odd"]) # if we guessed all odd
print ("Baseline (guess all \"odd\")")
print("Total Samples: ",len(training_data)," Total Correct: ",baseline_correct, "Accuracy: %.1f"%
     (float(baseline_correct/len(training_data))*100),"%")
print ("="*60)
print ("Naive Bayes Classifier")
print("Total Samples: ",len(training_data)," Total Correct: ",correct, "Accuracy: %.1f"%
     (float(correct/len(training_data))*100),"%")
print ("="*60)



