import itertools as it
from tqdm import tqdm
import math


K = 15
print('Number of potential solutions', math.factorial(15))
points = list(range(K))
print(points)
#permutations = it.permutations(points, r = K)
#sum([1 for _ in tqdm(permutations)]) # 1,307,674,368,000





import numpy as np
import matplotlib.pyplot as plt
import random



def gen_locations(seed=123, size=15):
    '''
    Generate (size=15) random locations
    '''
    np.random.seed(seed)
    x = np.random.randint(low = 1, high = 10+1, size = size)
    y = np.random.randint(low = 1, high = 10+1, size = size)
    return x,y



def distance(p1, p2):
    '''Returns Euclidean Distance between two points'''
    x1, y1 = p1
    x2, y2 = p2
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if d == 0:
        d = np.inf
    return d

def path_cost(path):
    start = path.pop() ## should be in order
    d = 0 # distance
    for p2 in reversed(path):
        d += distance(start, p2)
        print('{} --> {}: {}'.format(start, p2, round(d, 3)))
        start = p2
    return d

x,y = gen_locations(456,15) ## lowest distance 32.85
points = list(zip(x,y))

path_cost(points) # not accounting the trip back!

plt.scatter(x,y) ## lowest distance 32.85





class Path:
    def __init__(self, points):
        self.points = points # genes
        self._create_path() # ordered set of points
    
    def __repr__(self):
        return 'Path Distance: {}'.format(round(self.distance, 3))
    
    def __len__(self):
        return len(self.points)
    
    def copy(self):
        return Path(points = self.points)
        
    def _create_path(self, return_=False):
        points = self.points.copy()
        _init = points.pop(0) # get first item as start
        p1 = tuple(_init) # make copy; it will be end as well
        path = [p1]
        d = 0 # initial distance
        for p2 in points:
            path.append(p2) # save the points
            d += distance(p1, p2) # update the distance
            p1 = p2 # update current location 
        path.append(_init) ## start == end
        d += distance(p1, _init)
        self.path = path
        self.distance = d
        
        if return_:
            return d, path # return distance and path
    
    def plot(self, i=''):
        x,y = list(zip(*self.points))
        plt.scatter(x, y, marker='x')
        a,b = list(zip(*self.path))
        plt.plot(a,b)
        plt.title('{} Distnace: {}'.format(i,round(self.distance, 3)))



x,y = gen_locations(456,15) ## lowest distance 32.85
points = list(zip(x,y))
p = Path(points)
print('Initial', p)

p.plot() # initial random potential route



class GreedySolve:
    def __init__(self, x, y):
        '''
        Greedy solution for Traveling Salesman Probmem
        
          The main idea behind a greedy algorithm is local optimization. 
          That is, the algorithm picks what seems to be the best thing to do at
          the particular time, instead of considering the global situation.
          Hence it is called "greedy" because a greedy person grabs anything 
          good he can at the particular time without considering the long-range 
          implications of his actions.
        '''
        self.x = x
        self.y = y
        self.path = None # (np.array, np.array)
    
    def plot(self):
        plt.scatter(self.x, self.y)
        if self.path:
            a,b = list(zip(*self.path))
            plt.plot(a,b)
            plt.title('Distnace: {}'.format(self.distance))

    def solve(self, return_=False):
        '''Greedy iterative solution to the TP problem from (0,0)'''
        points = list(zip(self.x, self.y))
        _init = points.pop() # choose a point to begin the path
        # starting point shouldn't matter since it's a hamiltonian graph
        start = tuple(_init) # copy
        d = 0
        path = [start]
        iters = range(len(points))
        for _ in tqdm(iters):
            dist, p2 = _return_closest(start, points)
            points.remove(p2)
            path.append(p2)
            d += dist
            start = p2
        d += distance(start, _init)
        path.append(_init)
        
        #print(path)
        self.path = path
        self.distance = round(d, 3)
        
        if return_:
            return round(d,3), path

        
def _return_closest(p1, candidates):
    distances = []
    for p2 in candidates:
        d = distance(p1, p2)
        distances.append((d, p2))
    distances.sort(key = lambda x: x[0], reverse=False)
    return distances[0] ## return (distance, p2)
    



solution = GreedySolve(x, y)
solution.solve()
solution.plot()



path_cost(solution.path.copy())





class SA:
    def __init__(self, points, T):
        '''
        Simulated Algorithm solver for TSP
        '''
        self.points = random.sample(population=points, k=len(points)) ## reshuffle the cities
        self.T = T
        #self.func = func
    
    def solve(self, N=50):
        T = self.T
        path = Path(self.points)
        plt.ion()
        iterations = np.arange(N)
        fitness = []
        best_path = path.copy()
        for it in iterations:
            points = self._rearrange(path.points.copy())
            prop = Path(points)
            hprop = prop.distance
            hcur = path.distance
            p = max(0, min(1, np.exp(-(hprop - hcur) / T)))
            if hcur > hprop: ## suggested by Zbigniew Michaelwics and David Fogel (How to Solve It: Modern Heuristics)
                p = 1
            if np.random.rand() < p:
                path = prop
            T = 0.95 * T
            fitness.append(path.distance)
            if best_path.distance > path.distance:
                best_path = path.copy()
            if it % 100 == 0 :
                plt.cla()
                path.plot('{} with p={}'.format(it, round(p,3)))
                plt.pause(0.15)
        plt.cla()
        best_path.plot()
        plt.pause(5)
        plt.cla()
        plt.plot(iterations, fitness)
        plt.xlabel('iterations')
        plt.ylabel('distance')
        plt.title('distance over time (lower is better)')
        plt.show()
        plt.pause(5)
        self.T = T
        return best_path
    
    def _rearrange(self, points):
        '''
        Edit the path by taking a section of the path, reversing the order and placing it back in
        e.g. path: 1,2,3,4,5,6,7,8,9,10 --> [4,3,2,1],5,6,7,8,9,10
        '''
        stop = random.randint(1, len(points))
        start = random.randint(0, stop)
        points[start:stop] = reversed(points[start:stop])
        return points



solver = SA(points = points, T = 125)
solution = solver.solve(N=1200)

solution.plot()
print('Solution', solution, 'final temp', round(solver.T, 6))
print(str(solution.path))
plt.show()











