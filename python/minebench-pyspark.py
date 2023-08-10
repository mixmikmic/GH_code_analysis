import pyspark
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()
    
# Load the CSV as a Spark DataFrame
blocks_df = (spark.read
                  .option("header", "false")
                  .option("mode", "DROPMALFORMED")
                  .csv("100_blocks.csv"))

blocks_df.show()

from minebench import Minebench, InputUtils, FormatUtils


# Re-mine the blocks
hashes_rdd = blocks_df.rdd.map(lambda row: Minebench.get_block_header(row,
                                                                      bits=0x1FFFFFFF,
                                                                      sequential_nonce=True).mine())

hashes_rdd.collect()

