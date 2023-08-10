import Orange  # Needed package
import Orange.data
from orangecontrib.associate.fpgrowth import *  

data = Orange.data.Table("market-basket") # Orange data object for association rules

data

X, mapping = OneHot.encode(data, include_class=True)

X

sorted(mapping.items())

help(OneHot.decode)

names = {item: '{}'.format(var)
         for item, var, _ in OneHot.decode(mapping, data, mapping)}

names

class_items = {item
               for item, var, _ in OneHot.decode(mapping, data, mapping)}
sorted(class_items)

itemsets = dict(frequent_itemsets(X, .3)) # Find all itemsets with at least 30% support

rules = association_rules(itemsets, .8) #Generate association rules from these itemsets with minimum 80% confidence

rules_list = list(rules)

rules_list

# ante-> antecedent ; cons-> consequent ; supp -> support ; conf -> confidence
for ante, cons, supp, conf in rules_list:
    print('(supp: {}, conf: {})'.format(supp/len(X), conf),'  ',', '.join(names[i] for i in ante),'-->',names[next(iter(cons))])

itemsets_1 = dict(frequent_itemsets(X, .5)) # Find all itemsets with at least 50% support

rules_1 = association_rules(itemsets_1, .8) #Generate association rules from these itemsets with minimum 80% confidence

rules_list_1 = list(rules_1)

rules_list_1

for ante, cons, supp, conf in rules_list_1:
    print('(supp: {}, conf: {})'.format(supp/len(X), conf),'  ',', '.join(names[i] for i in ante),'-->',names[next(iter(cons))])

