import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#Utility functions
def getUnusedCoordinate(numCoords,usedCoordinates=[]):
    while True:
        coordinate = random.randint(1,numCoords)
        if coordinate not in usedCoordinates:
            return coordinate
        
        
def placeShips(numShips,numCoords):
    shipLocation = [0]*numShips
    
    for i in range(numShips):
        shipLocation[i] = getUnusedCoordinate(numCoords,shipLocation)
    return shipLocation

def playSimpleBattleshipAttacker(numGames,numCoordinates,numShips):
    experimentResult = np.empty(NUM_EPISODES)

    #Pick a random place to place the ships 
    shipsCoordinate = placeShips(NUM_SHIPS, NUM_COORDINATES)

    for curEpisodeIdx in range(NUM_EPISODES):

        shipsSunk = [False]*NUM_SHIPS
        keepShooting = True
        numShotsTaken = 0
        shotsTaken = list()

        #Fire randomly until ship is sunken. 
        while keepShooting:
            randomShotCoordinate = random.randint(1,NUM_COORDINATES)

            #If we have shot this before, try again. 
            if randomShotCoordinate in shotsTaken:
                continue
            else:
                shotsTaken.append(randomShotCoordinate)
                numShotsTaken += 1

                for curShip in range(NUM_SHIPS):
                    if shipsCoordinate[curShip] == randomShotCoordinate:
                        shipsSunk[curShip] = True

                if all(shipSunken==True for shipSunken in shipsSunk):
                    keepShooting = False
                    experimentResult[curEpisodeIdx]=numShotsTaken
    return experimentResult

NUM_EPISODES = 10000
NUM_COORDINATES = 100
NUM_SHIPS = 1

experimentResult = playSimpleBattleshipAttacker(NUM_EPISODES,NUM_COORDINATES,NUM_SHIPS)

plt.hist(experimentResult,bins=NUM_COORDINATES,normed=True,range=(1,NUM_COORDINATES));
plt.title('PDF of number of shots required to sink ONE ship')
plt.xlabel('Number of shots')
plt.ylabel('Probability')

avgNumberOfShotsToWIn = experimentResult.mean()
plt.axvline(x=avgNumberOfShotsToWIn,color="red")

print("The mean is: " + str(avgNumberOfShotsToWIn))

NUM_EPISODES = 10000
NUM_COORDINATES = 100
NUM_SHIPS = 2

experimentResult = playSimpleBattleshipAttacker(NUM_EPISODES,NUM_COORDINATES,NUM_SHIPS)

plt.hist(experimentResult,bins=NUM_COORDINATES,normed=True,range=(1,NUM_COORDINATES));
plt.title('PDF of number of shots required to sink TWO ships')
plt.xlabel('Number of shots')
plt.ylabel('Probability')

avgNumberOfShotsToWIn = experimentResult.mean()
plt.axvline(x=avgNumberOfShotsToWIn,color="red")

print("The mean is: " + str(avgNumberOfShotsToWIn))

NUM_EPISODES = 10000
NUM_COORDINATES = 100
NUM_SHIPS = 3

experimentResult = playSimpleBattleshipAttacker(NUM_EPISODES,NUM_COORDINATES,NUM_SHIPS)

plt.hist(experimentResult,bins=NUM_COORDINATES,normed=True,range=(1,NUM_COORDINATES));
plt.title('PDF of number of shots required to sink THREE ships')
plt.xlabel('Number of shots')
plt.ylabel('Probability')

avgNumberOfShotsToWIn = experimentResult.mean()
plt.axvline(x=avgNumberOfShotsToWIn,color="red")

print("The mean is: " + str(avgNumberOfShotsToWIn))

NUM_EPISODES = 10000
NUM_COORDINATES = 100

#when playing with random shots, have 17 ships occupying 1 
#square is the same as having multiple ships occupying 17 squares
#in total
NUM_SHIPS = 17

experimentResult = playSimpleBattleshipAttacker(NUM_EPISODES,NUM_COORDINATES,NUM_SHIPS)

fig = plt.hist(experimentResult,bins=NUM_COORDINATES,normed=True,range=(1,NUM_COORDINATES));
plt.title('PDF of number of shots required to sink 17 ships')
plt.xlabel('Number of shots')
plt.ylabel('Probability')

avgNumberOfShotsToWIn = experimentResult.mean()
plt.axvline(x=avgNumberOfShotsToWIn,color="red")

print("The mean is: " + str(avgNumberOfShotsToWIn))

