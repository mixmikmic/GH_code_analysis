#Set up a spark context

from pyspark import SparkContext,  SparkConf

conf = SparkConf().setAppName("implicitALS")
sc = SparkContext(conf=conf)

#Downloading and unzipping the data
get_ipython().system('wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip')
get_ipython().system('unzip BX-CSV-Dump.zip')

get_ipython().system('grep \'"0"\' BX-Book-Ratings.csv > implicit.csv')

#Load in the data
#The data is csv, with ';' as a delimiter, hence the split command. 
#The data has quote marks around all info, so I remove these with a replace mapping. 
#The first bit of data is user id, the second is the book isbn number, 
# and the third is the observation. 
ratings = sc.textFile('implicit.csv').map(lambda x: x.replace('"',""))             .map(lambda x:x.split(";"))            .map(lambda x:(int(x[0]), str(x[1]), int(x[2])))

ratings.take(10)

# Extract unique isbns.
isbns=ratings.map(lambda x:x[1]).distinct()
#Associates an integer with each unique isbn.
isbns_with_indices=isbns.zipWithIndex() 
#sets isbn as the key
reordered_ratings = ratings.map(lambda x:(x[1], (x[0], x[2]))) 
joined = reordered_ratings.join(isbns_with_indices) #joins with indexes 
joined.take(10)

#The data above is of the form :
    #(isbn, ((userid, rating), isbn-id-integer))
#We use the map function to get to the form :
    #(user id, isbn-id-integer, rating)
#This is the form expected by the ALS function
ratings_int_nice = joined.map(lambda x: (x[1][0][0], x[1][1], x[1][0][1]))
ratings_int_nice.take(10)

#Need 1s not 0s. since the matrix is singular if 0s. 
#i.e. we use '1' to indicate response, not 0.
ratings_ones = ratings_int_nice.map(lambda x:(x[0], x[1], 1))

from pyspark.mllib.recommendation import ALS
model=ALS.trainImplicit(ratings_ones, rank=5, iterations=3, alpha=0.99)

#Filter out all the  id of all books rated by user id = 8. 
users_books = ratings_ones.filter(lambda x: x[0] is 8).map(lambda x:x[1])
books_for_them = users_books.collect() #Collect this as a list

unseen = isbns_with_indices.map(lambda x:x[1])                             .filter(lambda x: x not in books_for_them)                             .map(lambda x: (8, int(x)))
unseen.take(10)

#Using the predict all function to give predictions for any unseens. 
predictions = model.predictAll(unseen)

predictions.take(10)

predictions.takeOrdered(20, lambda x: -x[2])

model.recommendProducts(8,10)

