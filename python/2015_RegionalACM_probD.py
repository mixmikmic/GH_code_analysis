def problem_D():
    T = int(raw_input().strip())
    for t in xrange(T):
        a = [int(i) for i in raw_input.strip().split()]
        scale = float(a[2])/a[1]
        ingredients = []
        for i in xrange(a[0]):
            name, weight, percent = raw_input.strip().split()
            if float(percent) == 100:
                main = i
            ingredients.append([name, weight, percent])
            
        scaled_main = ingredients[main][1]*scale
        print "Recipe #", t
        for i in xrange(ingredients):
            print ingredients[i][0], float(ingredients[i][2])*scaled_main

def probD():
    T=int(raw_input())
    for i in range(T):
        places=[]
        n=int(raw_input())
        for j in range(n):
            new=str(raw_input())
            if new not in places:
                places.append(new)
        print len(places)

def recipies():
    """scaling factor = desired portions/original number of portions
        main scaled weight = weght of main * scaling factor
        other ingredients = main scaled weight * other Baker's percentage"""

    test_cases = input()
    for i in xrange(testcases):
        num_ingredients = input()
        original_portions = input()
        desired_portions = input()
        print "Recipe #", i
        for j in xrange(num_ingredients):
            name = input()
            weight = input()
            percentage = input()

def problem_d():
    cases = int(raw_input())#how many cases
    for k in xrange(cases):

        okay = raw_input().split()
        ingredients = []
        weights = []
        percentages = []
        main = []
        #new_stuff = []
        scaling_factor = float(okay[2]) / float(okay[1])
        for j in xrange(int(okay[0])):
            this = raw_input().split()
            if this[2] == "100.0":
                main.append(this[0])
                main.append(this[1])
                main.append(this[2])
            ingredients.append(this[0])
            weights.append(this[1])
            percentages.append(this[2])
        print "Recipe # " + str(k+1)
        scaled_weight = float(main[1]) * scaling_factor
        for i in xrange(len(ingredients)):
            here = float(scaled_weight) * float(percentages[i]) / 100.0
            print ingredients[i] + " " + str(here)
            #new_stuff.append(here)

def ProbD():
    T = int(raw_input())
    for recipe in xrange(T):
        IN = map(int, raw_input().strip().split(' '))
        R = IN[0]
        P = IN[1]
        D = IN[2]
        L = []
        scale = float(D) / P
        main_weight = 0
        for i in xrange(R):
            new_ingredient = raw_input().strip().split(' ')
            ingredient = Ingredient(new_ingredient[0], new_ingredient[1], new_ingredient[2])
            L.append(ingredient)
            if ingredient.isMain():
                ingredient.weight = ingredient.weight*scale
                main_weight = ingredient.weight

        print "Recipe # " + str(recipe + 1)
        for ingredient in L:
            ingredient.weight = ingredient.percentage * main_weight / 100.0
            print ingredient
        print "----------------------------------------"

# Problem D
def recipes():
    for t in xrange(T):
        I = map(int,raw_input().strip().split())
        ingredients = []
        factor = float(I[2])/I[1]
        main = 0
        main_weight = 0
        for i in xrange(I[0]):
            ingredients.append(map(str,raw_input().strip().split()))
            if float(ingredients[i][2]) == 100.0:
                main = float(ingredients[i][1])
                main_weight = main * factor
        print "Recipe # " + str(t + 1)
        for i in xrange(len(ingredients)):
            ingredients[i][1] = float(ingredients[i][2]) * main_weight *.01
            print(ingredients[i][0]), "%.1f" % ingredients[i][1]
        print "-"*40

# RECIPE PROBLEM
number_of_test_cases = input()
for i in range(number_of_test_cases):
    number_of_ingredients, original_portion, new_portion = raw_input().strip().split()
    recpie = {}
    main_weight = 0
    for n in range(int(number_of_ingredients)):
        name, weight, percentage = raw_input().strip().split()
        
        recpie[name] = (float(weight), float(percentage))
        scaling_factor = float(new_portion)/float(original_portion)
        if float(percentage) == 100.0:
            main = name
            main_weight = recpie[main][0] * scaling_factor
    for key,value in recpie.items():
        print key +": "+ str(value[1]*main_weight/100)

#Problem D
import numpy as np

class Ingredient(object):

    def __init__(self, name, weight, percentage):
        self.name = name
        self.weight = float(weight)
        self.percentage = float(percentage)
        if float(percentage) == 100.0:
            self.main = True
        else:
            self.main = False

    def isMain(self):
        return self.main

    def __str__(self):
        ret_str = ""
        ret_str += self.name + " " + str(self.weight)
        return ret_str


def ProbD():
    T = int(raw_input()) # .strip()
    for recipe in xrange(T):
        IN = map(int, raw_input().strip().split(' '))
        R = IN[0]
        P = IN[1]
        D = IN[2]
        L = []
        scale = float(D) / P
        main_weight = 0
        for i in xrange(R):
            new_ingredient = raw_input().strip().split(' ')
            ingredient = Ingredient(new_ingredient[0], new_ingredient[1], new_ingredient[2])
            L.append(ingredient)
            if ingredient.isMain():
                ingredient.weight = ingredient.weight*scale
                main_weight = ingredient.weight
                
        print "Recipe # " + str(recipe + 1)
        for ingredient in L:
            ingredient.weight = ingredient.percentage * main_weight / 100.0
            print ingredient
        print "----------------------------------------"



