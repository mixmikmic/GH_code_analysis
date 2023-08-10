import bs
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

endReturns = lambda u, d, N, initial: [((u**k)*(d)**(N-k))*initial for k in range(0, N+1)]
returns = endReturns(1.02,0.99,2,100)

backwards = np.array([max(returns[i]-99, 0.) for i in range (0, 2+1)])

print(backwards)
final = lambda pstar, arr: [arr[0+i]*(1-pstar)+arr[1+i]*pstar for i in range(0, len(arr)-1)]
p = (1-0.99)/(1.02-0.99)

for i in range (0, 2):
    backwards = final(p, backwards)
    print (backwards)
    

def shark(spot, strike, tau, rate, vola, steps):
    """Fake numerical model by using noise."""
    #ref = bs.bscall(spot, strike, tau, rate, vola)
    
    endReturns = lambda ul, dl, N, initial: [((ul**k)*(dl)**(N-k))*initial for k in range(0, N+1)]
    final = lambda pstar, arr, r, t: [np.exp(-r*t)*(arr[0+i]*(1-pstar)+arr[1+i]*pstar) for i in range(0, len(arr)-1)]
    
    deltaT = tau/steps
    u = np.exp(vola * np.sqrt(deltaT))
    d = 1/u
    p = (np.exp(rate * deltaT) - d)/(u-d)
    discountFactor = np.exp(rate * deltaT)
    
    finalValue = endReturns(u,d,steps,spot)
    pStar = (discountFactor-d)/(u-d)
    
    BackwardLast = np.array([max(finalValue[i]-strike, 0.) for i in range(0, steps+1)])
    for i in range(0, steps):
        BackwardLast = final(pStar, BackwardLast, rate, deltaT)
        
    return BackwardLast

from config import *
OPTION

spot = 95.0
ref = bs.bscall(spot, OPTION["strike"], OPTION["tau"], OPTION["rate"], OPTION["vola"])
ref

ref - shark(spot, OPTION["strike"], OPTION["tau"], OPTION["rate"], OPTION["vola"], 500)

steps = range(10, 25,25)
prices = [shark(spot, OPTION["strike"], OPTION["tau"], OPTION["rate"], OPTION["vola"], n) for n in steps ]

plt.grid()
plt.plot(steps, prices-ref)

from tradgame import *
shark30 = partial(shark, 30)
shark100 = partial(shark, 100)

from UnitTest import RunTests
RunTests(bs.bscall) # this obviously passes, try it with RunTests(shark30) to see it fail

gameParameters = {"seed" : 932748239, "quotewidth" : 0.3, "delay" : 0.1, "steps" : 200}
game = TradingGame(gameParameters)
game.run([shark30, shark100])
game.ranking()

EventPlot(game).plot()

get_ipython().run_line_magic('time', 'shark(spot, OPTION["strike"], OPTION["tau"], OPTION["rate"], OPTION["vola"], 100)')

