import numpy as np
dataset_filename = "affinity_dataset.txt"
X = np.loadtxt(dataset_filename)

n_samples, n_features = X.shape
print(X[:5])
print("\nn_samples={}".format(n_samples))
print("n_features={}".format(n_features))
features = ["bread", "milk", "cheese", "apple", "banana"]
BREAD = 0
MILK = 1
CHEESE = 2
APPLE = 3
BANANA = 4

# How many rows contains our premise?
num_apple_purchases = 0
for sample in X:
    bought_apple = sample[APPLE] == 1
    if bought_apple:
        num_apple_purchases += 1
print("{0} people bought apples".format(num_apple_purchases))

# Find out how many users that bought apples bought banana too
valid_rule = 0
invalid_rule = 0
for sample in X:
    bought_apple = sample[APPLE] == 1
    bought_banana = sample[BANANA] == 1
    if bought_apple:
        if bought_banana:
            valid_rule += 1
        else:
            invalid_rule += 1

print("valid_rules={}".format(valid_rule))
print("invalid_rules={}".format(invalid_rule))

# Calculate Support and Confidence
support = valid_rule
confidence = valid_rule / num_apple_purchases

print("The support is {} and the confidence is {:.3f}.".format(support, confidence))

from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences = defaultdict(int)

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue

        # Increment the number of occurences
        num_occurences[premise] += 1
        
        for conclusion in range(n_features):
            if premise == conclusion: 
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise, conclusion)] += 1

support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]
    
def print_report(feature_a, feature_b, support, confidence):
    print("Rule: If a person buys {}, they will also buy {}.".format(feature_a, feature_b))
    print("- Support: {}".format(support))
    print("- Confidence: {:.3f}\n".format(confidence))

for (premise, conclusion) in confidence:
    print_report(features[premise], features[conclusion], support[(premise, conclusion)], confidence[(premise, conclusion)])

from operator import itemgetter

sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

for item in sorted_support[:5]:
    (premise, conclusion) = item[0]
    print_report(features[premise], features[conclusion], support[(premise, conclusion)], confidence[(premise, conclusion)])

from operator import itemgetter

sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for item in sorted_confidence[:5]:
    (premise, conclusion) = item[0]
    print_report(features[premise], features[conclusion], support[(premise, conclusion)], confidence[(premise, conclusion)])

