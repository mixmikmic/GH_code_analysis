
# Load in the functions
from databaker.framework import *

# Load the spreadsheet
tabs = loadxlstabs("example1.xls")

# Select the first table
tab = tabs[0]

print("The unordered bag of cells for this table looks like:")
print(tab)

# Preview the table as a table inline
savepreviewhtml(tab)

bb = tab.is_bold()
print("The cells with bold font are", bb)

print("The", len(bb), "cells immediately below these bold font cells are", bb.shift(DOWN))

cc = tab.filter("Cars")
print("The single cell with the text 'Cars' is", cc)

cc.assert_one()   # proves there is only one cell in this bag

print("Everything in the column below the 'Cars' cell is", cc.fill(DOWN))

hcc = tab.filter("Cars").expand(DOWN)
print("If you wanted to include the 'Cars' heading, then use expand", hcc)

print("You can print the cells in row-column order if you don't mind unfriendly code")
shcc = sorted(hcc.unordered_cells, key=lambda Cell:(Cell.y, Cell.x))
print(shcc)

print("It can be easier to see the set of cells coloured within the table")
savepreviewhtml(hcc)

"All the cells that have an 'o' in them:", tab.regex(".*?o")

# We get the array of observations by selecting its corner and expanding down and to the right
obs = tab.excel_ref('B4').expand(DOWN).expand(RIGHT)
savepreviewhtml(obs)

# the two main headings are in a row and a column
r1 = tab.excel_ref('B3').expand(RIGHT)
r2 = tab.excel_ref('A3').fill(DOWN)

# here we pass in a list containing two cell bags and get two colours
savepreviewhtml([r1, r2])

# HDim is made from a bag of cells, a name, and an instruction on how to look it up 
# from an observation cell.  
h1 = HDim(r1, "Vehicles", DIRECTLY, ABOVE)

# Here is an example cell
cc = tab.excel_ref('C5')

# You can preview a dimension as well as just a cell bag
savepreviewhtml([h1, cc])

# !!! This is the important look-up stage from a cell into a dimension
print("Cell", cc, "matches", h1.cellvalobs(cc), "in dimension", h1.label)

# You can start to see through to the final result of all this work when you 
# print out the lookup values for every observation in the table at once.  
for ob in obs:
    print("Obs", ob, "maps to", h1.cellvalobs(ob))

# You can change an output value like this:
h1.AddCellValueOverride("Cars", "Horses")

for ob in obs:
    print("Obs", ob, "maps to", h1.cellvalobs(ob))

# Alternatively, you can override by the reference to a single cell to a value 
# (This will work even if the cell C3 is empty, which helps with filling in blank headings)
h1.AddCellValueOverride(tab.excel_ref('C3'), "Submarines")
for ob in obs:
    print("Obs", ob, "maps to", h1.cellvalobs(ob))

# You can override the header value for an individual observation element.  
b4cell = tab.excel_ref('B4')
h1.AddCellValueOverride(b4cell, "Clouds")
for ob in obs:
    print("Obs", ob, "maps to", h1.cellvalobs(ob))

# The preview table shows how things have changed
savepreviewhtml([h1, obs])

wob = tab.excel_ref('A1')
print("Wrong-Obs", wob, "maps to", h1.cellvalobs(wob), " <--- ie Nothing")

h1.AddCellValueOverride(None, "Who knows?")
print("After giving a default value Wrong-Obs", wob, "now maps to", h1.cellvalobs(wob))

# The default even works if the cell bag set is empty.  In which case we have a special 
# constant case that maps every observation to the same value
h3 = HDimConst("Category", "Beatles")
for ob in obs:
    print("Obs", ob, "maps to", h3.cellvalobs(ob))


dimensions = [ 
    HDim(tab.excel_ref('B1'), TIME, CLOSEST, ABOVE), 
    HDim(r1, "Vehicles", DIRECTLY, ABOVE), 
    HDim(r2, "Name", DIRECTLY, LEFT), 
    HDimConst("Category", "Beatles")
]

c1 = ConversionSegment(obs, dimensions, processTIMEUNIT=False)
savepreviewhtml(c1)

# If the table is too big, we can preview it in another file is openable in another browser window.
# (It's very useful if you are using two computer screens.)
savepreviewhtml(c1, "preview.html", verbose=False)

print("Looking up all the observations against all the dimensions and print them out")
for ob in c1.segment:
    print(c1.lookupobs(ob))

df = c1.topandas()
df

print(writetechnicalCSV(None, c1))

# This is how to write to a file
writetechnicalCSV("exampleWDA.csv", c1)

# We can read this file back in to a list of pandas dataframes
dfs = readtechnicalCSV("exampleWDA.csv")
print(dfs[0])

# See that the `2014` no longer ends with `.0`
c1 = ConversionSegment(obs, dimensions, processTIMEUNIT=True)
c1.topandas()



