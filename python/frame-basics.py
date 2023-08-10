# First, let's verify that the SparkTK libraries are installed
import sparktk
print "SparkTK installation path = %s" % (sparktk.__path__)

from sparktk import TkContext
tc = TkContext()

# Create a new frame by providing data and schema
data = [ ['a', 1], 
         ['b', 2], 
         ['c', 3], 
         ['b', 4],     
         ['a', 5] ]

schema = [ ('letter', str),
           ('number', int) ]

frame = tc.frame.create(data, schema)

# View the first few rows of a frame
frame.inspect()

# View a specfic number of rows of a frame
frame.inspect(2)

# Add a column to the frame
frame.add_columns(lambda row: row.number * 2, ('number_doubled', int))
frame.inspect()

# Get summary information for a column
frame.column_summary_statistics('number_doubled')

# Add a column with the cumulative sum of the number column
frame.cumulative_sum('number')
frame.inspect()

# Rename a column
frame.rename_columns({ 'number_doubled': "x2" })
frame.inspect()

# Sort the frame by column 'number' descending
frame.sort('number', False)
frame.inspect()

# Remove a column from the frame
frame.drop_columns("x2")
frame.inspect()

# Download a frame from SparkTK to pandas
pandas_frame = frame.to_pandas(columns=['letter', 'number'])
pandas_frame

# Calculate aggregations on the frame
results = frame.group_by('letter', tc.agg.count, {'number': [tc.agg.avg, tc.agg.sum, tc.agg.min] })
results.inspect()

# Count the number of rows satisfying a predicate
frame.count(lambda row: row.number > 2)

