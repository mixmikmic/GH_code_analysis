# Create RDD from a python list
wordsRDD = sc.parallelize(["fish", "cats", "dogs"])

# View object type
print(wordsRDD)

# Get number of objects in RDD
print(wordsRDD.count())

# Get the first object in the RDD
print(wordsRDD.first())

# Collect all the objects in the RDD from Spark back to driver (this notebook)
print(wordsRDD.collect())

# Read in Moby Dick text file (reads in one line per RDD entry)
mobyDickRDD = sc.textFile("./data/MobyDick.txt")

# Count how many lines are in RDD
print("There is a total of %d lines in Moby Dick" % mobyDickRDD.count())

# Take first 5 lines from RDD
mobyDickRDD.take(50)

# The number of paritions for a RDD can be specified when RDD is created
# This is automatically done if no parameter is passed
N_partitions = 4
team = sc.parallelize(["Al", "Ani", "Jackie", "Lalitha", "Mark", "Neil", "Nick", "Shirin"], N_partitions)
print(team.getNumPartitions())



