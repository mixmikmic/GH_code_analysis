from PIL import Image
from PIL import ImageOps
import csv
    
csvfile = open("Data/yourDataSet", 'rU') 
reader = csv.reader(csvfile, delimiter=',', quotechar='"')
data = []
for row in reader:
    newRow = []
    for elem in row:
        newRow.append(float(elem))
    data.append(newRow)
csvfile.close()

numRows = len(data)
numCols = len(data[0])
img = Image.new('RGB', (numRows, numCols), "black")
pixels = img.load()

dataValues = []

for i in range(img.size[0]):
    for j in range(img.size[1]):
        if data[i][j] != 99999:
            dataValues.append(data[i][j])
        if data[i][j] <= 0.0:
            pixels[i,j] = (0, 0, 0)
        elif data[i][j] <= 0.0:
            pixels[i,j] = (0, 0, 0)
        else:
            pixels[i,j] = (0, 0, 0)

print "Minimum data value is", min(dataValues)
print "Maximum data value is", max(dataValues)
            
ImageOps.mirror(img.rotate(270, expand=1)).show()



