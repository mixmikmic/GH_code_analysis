import sys
T = int(raw_input().strip())

for t in xrange(T):
    N = int(raw_input().strip())
    votes = []
    for n in xrange(N):
        votes.append(int(raw_input().strip()))
    total = sum(votes)
    highest = max(votes)
    candidate = 1 + votes.index(highest)
    if votes.count(highest) > 1:
        print "no winner"
    elif float(highest)/total > 0.5:
        print "majority winner " + str(candidate)
    else:
        print "minority winner " + str(candidate)

# popular_vote.py

import numpy as np
import sys

# get number of test cases
test_cases = int(raw_input().strip())
# make lists of all results
for i in range(test_cases):
    """ David's approach
    n = int(raw_input().strip())
    votes = []
    for i in xrange(1, n+1):
        votes.append(int(raw_input().strip())
    total_votes = sum(votes)
    # get max value and index of max value in list
    indices_ord = [j for j in reversed(np.argsort(votes))]
    sorted_votes = [i for i in reversed(sorted(votes))]
    if sorted_votes[0] == sorted_votes[1]:
        print sorted_votes[0], sorted_votes[1]
        print "no winner"
    elif sorted_votes[0] > sum(sorted_votes[1:]):
        print "majority winner", indices_ord[0]+1
    else:
        print "minority winner", indices_ord[0]+1
    """
    winner = np.arg_max(arr)
    max_votes = np.pop(winner)
    if max_votes in arr:
        return "no winner"
    # get sum of all values
    to_print = "majority "
    if (max_value / float(np.sum(arr))) < 0.5:
        to_print = "minority "
    print to_print + str(winner)

# Majority Winner

number_of_test_cases = input()

for j in range(number_of_test_cases):
    n = input()
    votes_for_candidate = [0]*n
    for i in range(n):
        votes_for_candidate[i] = input()
    total_votes = sum(votes_for_candidate)
    winner = max(votes_for_candidate)
    votes_for_candidate.remove(winner)
    if winner in votes_for_candidate:
        print "no winner"
    elif winner > total_votes/2:
        print "majority winner"
    else:
        print "minority winner"

#ACM problems
import numpy as np
import copy

def problem_a():
    cases = int(raw_input())#how many test cases
    for k in xrange(cases):
        candidates = int(raw_input())#how many candidates
        votes = []
        total = 0
        for j in xrange(candidates):
            vote = int(raw_input())
            votes.append(vote)
            total += vote
        highest = max(votes)
        if votes.count(highest) > 1:
            print "no winner"
        else:
            for i in xrange(candidates):
                if votes[i] == highest:
                    if votes[i] > total / 2:
                        print "majority winner " + str(i+1)
                    else:
                        print "minority winner " + str(i+1)

numberTC=input()
c=0

while c<numberTC:
    mylist=[]
    numberCan=input()
    d=0
    while d<numberCan:
        mylist.append(input())
        d+=1
    totalSum=sum(mylist)
    maxList=max(mylist)
    if mylist.count(maxList)>1:
        print "no winner"
    elif float(maxList)/totalSum>.5:
        print "majority winner ",mylist.index(maxList)+1
    else:
        print "minority winner ",mylist.index(maxList)+1
        
    c+=1

num_test = int(raw_input())
for i in range(num_test):
    #print("for i")
    num_cand = int(raw_input())
    votes=[]
    for j in range(num_cand):
        #print("for j")
        votes.append(int(raw_input()))
    big = max(votes)
    big_ind = votes.index(max(votes)) + 1
    total = sum(votes)
    i = 0
    same = False
    while i < len(votes)-1:
        #print("while i")
        if votes[i] == votes[i+1]:
            same = True
        else:
            #print("break")
            same = False
            break
    if total/2 >= big:
        print("miniority winner", big_ind)
    elif total/2 < big:
        print("majority winner", big_ind)
    elif same == True:
        print("no winner")
        
        """It's really close but not quite there."""

# PROBLEM A
"""
McKell Stauffer
"""

TestCases = int(raw_input())
for i in xrange(TestCases):
    NumCanidates = int(raw_input())
    List = []
    for k in xrange(NumCanidates):
        List.append(int(raw_input()))
    Max = max(List)
    Index = List.index(Max)
    List.remove(List[Index])
    Winner = "Majority Winner"
    for j in xrange(len(List)):
        if List[j] == Max:
            Winner = "No Winner"
    if Winner == "No Winner":
        print "No Winner"
    else:
        if Max <= sum(List):
            Winner = "Minority Winner"
        print Winner, (Index+1)

