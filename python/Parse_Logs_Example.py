get_ipython().system('cat DataSets/LogDB.out | head')

# Capture "-"
get_ipython().system('cat DataSets/LogDB.out | perl -pe \'s/^.*"(.*)" (.*)$/\\1,\\2/\' | head')

# Split "\1" into three groups separated by space
get_ipython().system('cat DataSets/LogDB.out | perl -pe \'s/^.*"(.*?) (.*)? (.*)?"(.*)$/\\1,\\2,\\4/\' | head')

# Split \4 into number groups separated by space
get_ipython().system('cat DataSets/LogDB.out | perl -pe \'s/^.*"(.*?) (.*)? (.*)?" (\\d+) (\\d+) (\\d+.\\d+)$/\\1,\\2,\\4,\\5,\\6/\' | head')

get_ipython().system('cat DataSets/LogDB.out | perl -pe \'s/^.*"(.*?) (.*)? (.*)?" (\\d+) (\\d+) (\\d+.\\d+)$/\\1,"\\2","\\4",\\5,\\6/\'> DataSets/LogDB.csv')

get_ipython().system('echo "method,path,code,size,latency" > DataSets/LogDB_head.csv')

get_ipython().run_cell_magic('bash', '', '\n# Add headder\ncat <(echo "method,path,code,size,latency") DataSets/LogDB.csv > DataSets/LogDB_full.csv\n\n# View Statistics\ncat DataSets/LogDB_full.csv  | csvstat')

# plot histogram

get_ipython().system('csvcut -c "latency" DataSets/LogDB_full.csv | feedgnuplot --line --histogram 0 --with boxes --terminal "dumb 80,40"')



