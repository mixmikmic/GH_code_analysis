"""
This function behaves the same as the built in map function (except that this materializes the iterator and the build in one doesn't)
"""
def standardMap(f, seq):
    return [f(x) for x in seq]

from random import shuffle
"""
This function randomizes the order of the data passed in and then returns the list the results from applying the 
passed in function to each element of the shuffled list.
"""
def randomMap(f, seq):
    rseq = [x for x in seq]
    shuffle(rseq)
    return standardMap(f, rseq)

"""
Show results of the three different map functions at our disposal.
"""
def tryDifferentMaps(f, seq):
    print("builtin: " + str([x for x in map(f, seq)]))
    print("standard: " + str(standardMap(f, seq)))
    print("random: " + str(randomMap(f, seq)))

f = lambda x: x + 1
tryDifferentMaps(f, range(1, 5))

"""
This function is equivalent to the built in reduce function
"""
def standardReduce(f, seq):
    curval = seq[0]
    for newval in seq[1:]:
        curval = f(curval, newval)
    return curval

"""
This function is similar to the built in reduce function, but it shuffles the sequence before operating on it.
This is comparable to the problem faced when working on multiple processes/computers in parallel.
"""
def randomReduce(f, seq):
    rseq = [x for x in seq]
    shuffle(rseq)
    return standardReduce(f, rseq)

from functools import reduce

"""
Show results of the three different reduce functions at our disposal.
"""
def tryDifferentReductions(f, seq):
    print("builtin: " + str(reduce(f, seq)))
    print("standard: " + str(standardReduce(f, seq)))
    print("random: " + str(randomReduce(f, seq)))

f = lambda x, y: x*y # a commutative function
tryDifferentReductions(f, range(2,6))

f = lambda x, y: pow(x,y) # a non-commutative function
tryDifferentReductions(f, range(2,6))

f = lambda x, y: x+" "+y # another non-commutative function
tryDifferentReductions(f, "this is a sentence".split())

"""
The simplest version of map reduce that I could show.  This version does not account for the shuffle step, which would be
necessary to do a hash join efficiently.
"""
def simpleMapReduce(mapFunc, reduceFunc, seq):
    return randomReduce(reduceFunc, randomMap(mapFunc, seq))
    

mapFunc = lambda x:x
reduceFunc = lambda x,y: x+y
simpleMapReduce(mapFunc, reduceFunc, range(1,10))

"""
This is similar to the simple map reduce algorithm above, but it breaks the data apart and does each subset on its 
own before recombining everything together at the end.
"""
def randomMapReduce(mapFunc, reduceFunc, seq):
    workingList = [] #this variable will hold intermediate results after doing the first round of reduction operations.
    
    #break the input sequence up and treat each operation as if it's happening in parallel.
    sizePerChunk = 4
    def chunks():
        for i in range(0, len(seq), sizePerChunk):
            yield seq[i:i+sizePerChunk]
            
    chunkedList = list(chunks()) #chunkedList now holds a list of lists.
    
    #run reduce on each of the chunks and store the results of those operations in workingList
    for i in range(len(chunkedList)):
        workingList.append(simpleMapReduce( mapFunc, reduceFunc, chunkedList[i]))
    
    #once the parallel operations have been done, do the last combination task.
    return randomReduce(reduceFunc, workingList)

mapFunc = lambda x:x
reduceFunc = lambda x,y: x+y
randomMapReduce(mapFunc, reduceFunc, range(1,10))

"""
This function accepts two sorted lists and the comparison operator used to sort them and efficiently combines them into a 
single sorted list
"""
def mergeLists (left, right, comparator):
    retval = []
    leftPointer = 0
    rightPointer = 0
    while leftPointer+rightPointer < len(left)+len(right):
        #print("leftPointer: %i, rightPointer: %i"%(leftPointer, rightPointer))
        if leftPointer > len(left)-1: #did we fall off the end on the left
            retval.append(right[rightPointer])
            rightPointer += 1
        elif rightPointer > len(right)-1: #did we fall off the end on the right
            retval.append(left[leftPointer])
            leftPointer += 1
        elif comparator(left[leftPointer], right[rightPointer]): #is the left value ">" the right value
            retval.append(left[leftPointer])
            leftPointer += 1
        else:
            retval.append(right[rightPointer]) #the left value is not ">" the right value
            rightPointer += 1
    return retval

mapFunc = lambda x:[x]
reduceFunc = lambda x,y: mergeLists(x, y, lambda x, y: x<y)
simpleMapReduce(mapFunc, reduceFunc, range(1,10))

mapFunc = lambda x:[x]
reduceFunc = lambda x,y: mergeLists(x, y, lambda x, y: x>y)
simpleMapReduce(mapFunc, reduceFunc, range(1,10))

#1. Write map reduce to count the letters in the sentence 'This is pretty cool.'
mapFunc = lambda x:len(x)
reduceFunc = lambda x,y: x+y
seq = 'This is pretty cool.'.split()
simpleMapReduce(mapFunc, reduceFunc, seq)

#2. Write map reduce to get the count of words of each length in the above sentence.

#this is a dictionary wrapper that is identical to dictionary except that, in the case of invalid references, this function
#returns 0 instead of throwing an error.
class default0Dict(dict):
    def __missing__(self, key):
        return 0
    
mapFunc = lambda x:default0Dict({len(x):1})
reduceFunc = lambda x,y: default0Dict({key:x[key]+y[key] for key in #take the sum of the values in the two dictionaries for
                                                                   #this key
                                       set(x.keys()) | set(y.keys())} #take the union of the list of keys from both dicts
                                                                    #to determine which keys to have in the new dict
                                     )
seq = 'This is pretty cool.'.split()
randomMapReduce(mapFunc, reduceFunc, seq)

