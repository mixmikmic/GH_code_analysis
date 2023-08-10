import pymongo
from simhash_db import Client as Simdbclient

# TODO: the simhash module in this repo is *not* the source simhash module
#       but one maintained by moz (it contains the Corpus, see exception)

client = Simdbclient('mongo', name='testing', num_blocks=6, num_bits=1, host=['localhost'])




