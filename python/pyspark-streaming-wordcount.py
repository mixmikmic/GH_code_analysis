import pyspark
import time
from pyspark.streaming import StreamingContext
from socket import *
from threading import Thread

# Create a SparkContext with 2 threads in local mode
sc = pyspark.SparkContext("local[2]")

# Create a thread that reads streaming data from the socket and 
# performs a wordcount on the data that arrived in the last 1 second
class streamer(Thread):
    def __init__(self, sc):
        Thread.__init__(self)
        self.sc = sc

    def run(self):
        print("starting streamer thread")
        batchInterval = 1
        # Using the spark context, create a streaming context with a batch interval of 1 second
        ssc = StreamingContext(self.sc, batchInterval )
        # Create a socket DStream reading from localhost at port 4444
        socketDstream = ssc.socketTextStream("localhost",4444)
            
        # WordCount
        wordcounts = socketDstream.flatMap(lambda line: line.split(" "))                         .map(lambda word: (word, 1))                         .reduceByKey(lambda a, b: a+b)  
         
        # Print first 50 words counted in the last one second
        wordcounts.pprint(50)
        
        # Start the execution
        ssc.start()
        
        # Stop the streaming context after 35 seconds
        time.sleep(35)       
        ssc.stop()        

# Create a thread that writes data to a socket to simulate streaming data at regular intervals
class worker(Thread):
    def run(self):
        print('starting Socket writing thread')
        HOST = 'localhost'
        PORT = 4444
        ADDR = (HOST, PORT)
        # Create an INET, Streaming socket
        serversocket = socket(AF_INET, SOCK_STREAM)
        # Bind the socket to localhost and port 4444
        serversocket.bind(ADDR) 
        # Become a server socket
        serversocket.listen(SOMAXCONN)   
        # Accept connections
        clientsocket, addr = serversocket.accept()
        print ("connection ready.Sending data")
        
        # Text copied from Wikipedia: https://en.wikipedia.org/wiki/Russula_virescens 
        # Start sending one line of text every two seconds
        clientsocket.send("Russula virescens is a basidiomycete mushroom of the genus Russula and is commonly known as the green-cracking russula.\n\r")
        time.sleep(2)
        clientsocket.send("It can be recognized by its distinctive pale green cap, the surface of which is covered with darker green angular patches.\n\r")
        time.sleep(2)       
        clientsocket.send("It has crowded white gills, and a firm, white stipe that is up to 8 cm tall and 4 cm thick.\n\r")
        time.sleep(2)
        clientsocket.send("Considered to be one of the best edible mushrooms of the genus Russula, it is especially popular in Spain and China.\n\r")
        time.sleep(2)
        clientsocket.send("With a taste that is described variously as mild, nutty, fruity, or sweet, it is cooked by grilling, frying, sauteeing, or eaten raw.\n\r")
        time.sleep(2)
        clientsocket.send("Mushrooms are rich in carbohydrates and proteins, with a low fat content.\n\r")
        time.sleep(2)
        clientsocket.send("Its distribution encompasses Asia, North Africa, Europe, and Central America.\n\r")
        time.sleep(2)
        clientsocket.send("In Asia, it associates with several species of tropical lowland rainforest trees of the family Dipterocarpaceae.\n\r")
        time.sleep(2)
        clientsocket.send("R. virescens has a ribonuclease enzyme with a biochemistry unique among edible mushrooms.\n\r")
        time.sleep(2)
        clientsocket.send("It also has biologically active polysaccharides, and a laccase enzyme that can break down several dyes used in the laboratory and in the textile industry.\n\r")
        time.sleep(2)
        clientsocket.send("It may turn slightly brown with age, or when it is injured or bruised after handling.\n\r")
        time.sleep(2)
        clientsocket.send("The cap cuticle is thin and can be easily peeled off the surface to a distance of about halfway towards the cap center.\n\r")
        
        print("Socket writing thread exiting")
        
        
        

# Start the streamer thread to begin reading data from the socket and processing it
streamer(sc).start()

# Start the worker thread to begin writing data to the socket
worker().start()   
  



