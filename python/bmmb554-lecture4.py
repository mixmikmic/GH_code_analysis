def recMC(coinValueList,change):
   minCoins = change
   if change in coinValueList:
     return 1
   else:
      for i in [c for c in coinValueList if c <= change]:
         numCoins = 1 + recMC(coinValueList,change-i)
         if numCoins < minCoins:
            minCoins = numCoins
   return minCoins

print(recMC([1,5,10,25],63))

def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
   for cents in range(change+1):
      coinCount = cents
      newCoin = 1
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
               coinCount = minCoins[cents-j]+1
               newCoin = j
      minCoins[cents] = coinCount
      coinsUsed[cents] = newCoin
   return minCoins[change]

def printCoins(coinsUsed,change):
   coin = change
   while coin > 0:
      thisCoin = coinsUsed[coin]
      print(thisCoin)
      coin = coin - thisCoin

def main():
    amnt = 63
    clist = [1,5,10,21,25]
    coinsUsed = [0]*(amnt+1)
    coinCount = [0]*(amnt+1)

    print("Making change for",amnt,"requires")
    print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")
    print("They are:")
    printCoins(coinsUsed,amnt)
    print("The used list is as follows:")
    print(coinsUsed)

main()

import matplotlib.pylab as plt
import numpy as np

col_labels=['0','1','2','3','4']
row_labels= ['0','1','2','3','4']
table_vals= [
    ['0','3','5','9',''],
    ['', '', '', '13',''],
    ['','','','15','19'],
    ['','','','','20'],
    ['','','','','23']]
line = np.array([
    [0, 1, 2, 3, 3, 3, 4, 4, 4 ],
    [0, 0, 0, 0, 1, 2, 2, 3, 4 ]])    
ncol = len(col_labels)
nrow = len(row_labels)

# draw grid lines
plt.plot(np.tile([0, ncol+1], (nrow+2,1)).T, np.tile(np.arange(nrow+2), (2,1)),
    'k', linewidth=3)
plt.plot(np.tile(np.arange(ncol+2), (2,1)), np.tile([0, nrow+1], (ncol+2,1)).T,
    'k', linewidth=3)

# plot labels
for icol, col in enumerate(col_labels):
    plt.text(icol + 1.5, nrow + 0.5, col, ha='center', va='center')
for irow, row in enumerate(row_labels):
    plt.text(0.5, nrow - irow - 0.5, row, ha='center', va='center')

# plot table content
for irow, row in enumerate(table_vals):
    for icol, cell in enumerate(row):
        plt.text(icol + 1.5, nrow - irow - 0.5, cell, ha='center', va='center')

# plot line
plt.plot(line[0] + 1.5, nrow - line[1] - 0.5, 'r', linewidth = 5, alpha = 0.5)

plt.axis([-0.5, ncol + 1.5, -0.5, nrow+1.5])
plt.show()



