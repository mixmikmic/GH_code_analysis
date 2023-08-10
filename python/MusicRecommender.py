from pyspark.mllib.recommendation import *
import random
from operator import *

# Method to split artist Id and its name.
def splitArtistName(line):
    try:
        id, name = line.split("\t")
        return (int(id), name)
    except ValueError:
        return None

# Load text file where each line contains artist Id and its name.
artistData = sc.textFile("artist_data_small.txt")
# Split artist id: name and store in a map. 
artistData = artistData.map(splitArtistName).filter(lambda x: x!=None).collectAsMap()

'''
Load artist correct id and its aliases
    2 columns: badid, goodid
    known incorrectly spelt artists and the correct artist id. 
'''
artistAlias = sc.textFile('artist_alias_small.txt')
# Split Artist Alias data into (badId, goodId)
def splitArtistAlias(line):
    try:
        # Catches error in data
        badId, goodId = line.split("\t")
        return (int(badId), int(goodId))
    except ValueError:
        return None

# Create map badId: goodId

artistAlias = artistAlias.map(splitArtistAlias).filter(lambda x: x!=None).collectAsMap()

'''
Load data about user's music listening history
Each line contains three features: userid, artistid, playcount
'''
userArtistData = sc.textFile("user_artist_data_small.txt")

# Return the corrected user information.
def parseUserHistory(line):
    try:
        # Catch error in line
        user, artist, count = line.split()
        # Return the corrected user information.
        if artist in artistAlias:
            return (int(user), artistAlias[artist], int(count))
        else:
            return (int(user), int(artist), int(count))
    except ValueError:
        return None


# Create corrected user history RDD.
userArtistData = userArtistData.map(parseUserHistory)

userArtistData.cache()

userArtistPurge = userArtistData.map(lambda x: (x[0],x[2]))
# Create an RDD storing user information in the form of (total play count of all artists combined for the current user, (userId of the current user, number of unique artists listened by user))
songCountAgg = userArtistPurge.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1), lambda a,b: (a[0] + b[0], a[1] + b[1])).map(lambda x: (x[1][0], (x[0], x[1][1])))
# Sort the RDD based on the total play counts so as to find the most active user.
sortedCount = songCountAgg.sortByKey(False)
# Find the top 3 user information
sortedCountTop3 = sortedCount.take(3)

# Print the top 3 user information.

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[0][1][0],sortedCountTop3[0][0], sortedCountTop3[0][0]/sortedCountTop3[0][1][1])

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[1][1][0],sortedCountTop3[1][0], sortedCountTop3[1][0]/sortedCountTop3[1][1][1])

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[2][1][0],sortedCountTop3[2][0], sortedCountTop3[2][0]/sortedCountTop3[2][1][1])

trainData, validationData, testData = userArtistData.randomSplit([0.4,0.4,0.2], seed=100)

trainData.cache()
validationData.cache()
testData.cache()

def modelEval(model, data):
    artistsList =  broadcastVar.value
    total = 0.0
    userList = validationData.map(lambda x: x[0]).distinct().collect()
    for user in users:
        trainArtists = set(trainData.filter(lambda x: x[0]==userList).map(lambda x: x[1]).collect())
        #Remove artists for the current user in training Dataset from the userArtistData.
        nonTrainArtists = sc.parallelize([(user,artist) for artist in artistsList if not artist in trainArtists])
        #use the model to predict all the ratings on nonTrainArtists
        prediction = model.predictAll(nonTrainArtists)
        #top X sorted by highest rating from the prediction for the current user
        X = len(trueArtists)
        topX = sorted(prediction.collect(), key=lambda x: x.rating, reverse=True)[:X]
        
        trueArtists = set(data.filter(lambda x: x[0]==userList).map(lambda x: x[1]).collect())
        topArtist = set(topX.map(lambda x: x[1]))
        #Compare predictResult to trueArtists
        total += float(len(topArtist & trueArtists))/len(trueArtists)
    return total/len(userList)

allArtists = userArtistData.map(lambda x: x[1]).distinct().collect()
broadcastVar = sc.broadcast(allArtists)

# print the model accuracy score 
for val in [2, 10, 20]:
    model = ALS.trainImplicit(training, rank=val, seed=345)
    print("The model score for rank %d is %f" % (rank, modelEval(model, validationData)))

bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData)

top5=bestModel.recommendProducts(1059637, 5)
i=1
for val in top5:
    print "Artist %d : %s" %(i,dictionary[val[1]])
    i=i+1

