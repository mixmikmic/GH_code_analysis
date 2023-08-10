from databaker.framework import *

# put your input-output files here
inputfile = "example1.xls"
outputfile = "example1.csv"
previewfile = "preview.html"

from databaker.framework import *
tab = loadxlstabs("example1.xls", sheetids="stones", verbose=True)[0]
print(tab)

cellbag = tab
print("Numbered cells only:", cellbag.is_number())
print()
print("Not numbers:", cellbag.is_not_number())
print()
print("Not numbers and not whitespace:", cellbag.is_not_number().is_not_whitespace())
print()
print("Cells that seem to be a date:", cellbag.is_date())

from databaker.framework import *   # restated import so you can run from this cell
cellbag = tab = loadxlstabs("example1.xls", sheetids="stones", verbose=True)[0]

print("Get some matching cells", cellbag.one_of(["Rocks", "ice", "mud"]))
print("A3 is", cellbag.excel_ref("A3"))
print("A3:B4 is", cellbag.excel_ref("A2:B4"))
print()
print("The second cell in the whole table is", tab.by_index(2))

ngreater20 = cellbag.is_number().filter(lambda c: c.value>20)
nlessthan20 = cellbag.is_number().filter(lambda c: c.value<20)
print("Numbers greater than 20", ngreater20)
print("Numbers less than 20", nlessthan20)

# Uncomment this line to see these selections in contents
# savepreviewhtml([ngreater20, nlessthan20])

colC = tab.excel_ref("D3:D5")
rowC = tab.excel_ref("A4:D4")
print("colC", colC)
print("rowC", rowC)
print()
print("Union is", colC.union(rowC))
print("Difference is", colC.difference(rowC))
print("Intersection is", colC.intersection(rowC))
print()
print("Union is", (colC | rowC))
print("Difference is", (colC - rowC))
print("Intersection is", (colC & rowC))

c = tab.excel_ref("D3") | tab.excel_ref("E4")
d = tab.excel_ref("A6:A7")
print("Waffle:")
savepreviewhtml([c,d, c.waffle(d)])

print("Junction output:")
for s in c.junction(d):
    print("  ", s)

print("Cells column A that are in same row as", c, "are", tab.excel_ref("A").same_row(c))
print("Cells column 7 that are in same column as", c, "are", tab.excel_ref("7").same_col(c))

c = tab.excel_ref("B4")
print("Shift RIGHT from", c, "is", c.shift(RIGHT))
print("Shift (-1,-2) from", c, "is", c.shift((-1, -2)))
print("Fill UP from", c, "is", c.fill(UP))
print("Expand UP from", c, "is", c.expand(UP))
print()
print("How it works: UP=", UP, "  DOWN=", DOWN, "  LEFT=", LEFT, "  RIGHT=", RIGHT)
print()
print("Extrude two cells rightwards", c.extrude(2,0))

from databaker.framework import *
tab = loadxlstabs("example1.xls", sheetids="stones", verbose=False)[0]

rocks = tab.filter("Rocks").fill(DOWN)
years = tab.filter("Year").fill(DOWN).is_not_whitespace()
cost = tab.filter("cost").fill(DOWN)
print(rocks)

# savepreviewhtml([rocks, years, cost])  # <-- uncomment this line to see the table

hrocks = HDim(rocks, "ROCKS!", DIRECTLY, LEFT)
hrocks.AddCellValueOverride("granite", "gneiss")
hyears = HDim(years, "yyyy", CLOSEST, UP)

for ob in cost:
    print(ob, "\t", hyears.cellvalobs(ob), "\t", hrocks.cellvalobs(ob))

# savepreviewhtml([hrocks, hyears, cost])  # <-- uncomment to see as a coloured table

from databaker.framework import *

times = [2017.0, "Q32017", "Mar  2017"]
for t in times:
    print(t, "is\t", Ldatetimeunitloose(t), "corrected to\t", Ldatetimeunitforce(t, Ldatetimeunitloose(t)))

from databaker.framework import *
tab = loadxlstabs("example1.xls", sheetids="stones", verbose=False)[0]

cs = ConversionSegment(tab.filter("cost").fill(DOWN), [
        HDim(tab.filter("Year").fill(DOWN).is_not_whitespace(), "year", CLOSEST, UP),
        HDim(tab.filter("Month").fill(DOWN).is_not_whitespace(), "month", DIRECTLY, LEFT)
    ])


###################
# savepreviewhtml(cs)   # <-- uncomment this to see the interactive table

dcs = cs.topandas()
# print(dcs)   # uncomment to see the table

# concatenate the month and year into a time
dcs["TIME"] = dcs.month + " " + dcs.year
pdguessforceTIMEUNIT(dcs)   # <-- fixes the date format (removing the '.0's on the years)
# print(dcs)   # uncomment to see the table at this point

# delete the now redundant columns 
dcs.drop(['year', "month"], axis=1, inplace=True)
#print(dcs)  # uncomment to see pandas table

# Output the finished WDA file where the dates should all work!
print(writetechnicalCSV(None, dcs))

import urllib, re, os

# url containing the index of a set of spreadsheets
ddurl = "https://www.ons.gov.uk/businessindustryandtrade/constructionindustry/datasets/outputintheconstructionindustry/current"
req1 = urllib.request.Request(ddurl, headers={'User-Agent' : "Sensible code"}) 
dhtml = urllib.request.urlopen(req1).read().decode("utf8")
print("Downloaded a webpage with", len(dhtml), "bytes")

# make the download directory
dfiles = "downloaddir"
if not os.path.isdir(dfiles):
    print("making directory", dfiles)
    os.mkdir(dfiles)

# quick and dirty regular expression for pullint out the links to relevant xls spreadsheets
xllinklist = re.findall('href="(/file\?uri=/businessindustryandtrade.*?/([^/"]*\.xls))"', dhtml)
    
for xl, xln in xllinklist:
    lxln = os.path.join(dfiles, xln)
    if os.path.exists(lxln):
        continue   # <-- we avoid downloading the same file a second time, in this case
    furl = urllib.parse.urljoin(ddurl, xl)
    req = urllib.request.Request(furl, headers={'User-Agent' : "Sensible code"}) 
    xp = urllib.request.urlopen(req).read()
    print("Downloading", xln, len(xp), "bytes")
    fout = open(lxln, "wb")
    fout.write(xp)
    fout.close()

fnames = [ os.path.join(dfiles, f)  for f in os.listdir(dfiles)  if f[-4:] == '.xls' ]

print("Your list of xls files is:\n", "\n ".join(fnames))

import urllib, re

# fetch the front page and find the link to the zip file we want
iurl = "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/workplacepensions/datasets/annualsurveyofhoursandearningspensiontablespensiontypebyagegroupandbygrossweeklyearningsbandsp1"
req = urllib.request.Request(iurl, headers={'User-Agent' : "Sensible Code"}) 
ipage = urllib.request.urlopen(req).read()

# search the link to the zip file and "join" against the baseurl to get the full url (there's a space -> %20 bug problem)
zyears = [ urllib.parse.urljoin(iurl, z.replace(" ", "%20"))  for z in re.findall('<a href="([^"]*?\.zip)"', str(ipage)) ]
zurl = zyears[0]

print("We are about to download the file:\n", zurl)
zfilename = "downloaded.zip"
zurl = zurl.replace(" ", "%20")   # spaces in the url get escaped in the browser
req = urllib.request.Request(zurl, headers={'User-Agent' : "Sensible Code"}) 
zbytes = urllib.request.urlopen(req).read()
    
fout = open(zfilename, "wb")
fout.write(zbytes)
fout.close()
print(zfilename, "is", len(zbytes), "bytes long.")    

import zipfile

zfilename = "downloaded.zip"

# open the zipfile
zdir = zipfile.ZipFile(zfilename)

print("The files in", zfilename, "are:\n", "\n ".join(zdir.namelist()))

zmember0 = zdir.namelist()[0]

xlsfilename = "downloaded0.xls"
fout = open(xlsfilename, "wb")
xlsbindata = zdir.read(zmember0)
fout.write(xlsbindata)   
fout.close()

print()
print("We have unzipped:\n", zmember0, "\nand saved it as", xlsfilename, "with", len(xlsbindata), "bytes")

# now we can load this file into databaker and continue with our work
from databaker.framework import *
tabs = loadxlstabs(xlsfilename)



