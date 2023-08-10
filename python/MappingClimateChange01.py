from PIL import Image
from PIL import ImageOps
import csv


def main():
    stringReadin = readInCSVToString("LandTempDec15Big.csv")
    floatData = stringDataToFloat(stringReadin)
    MapClimate = makeAndFillImg(floatData)
    MapClimate.show()


def readInCSVToString(filepath):
    csvfile = open(filepath, 'rU') 
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    stringRead = []
    for row in reader:
        stringRead.append(row)
    csvfile.close()
    return stringRead


def stringDataToFloat(stringRead):
    finalRead = []
    for stringRow in stringRead:
        numRow = []
        for stringElem in stringRow:
            val = float(stringElem)
            numRow.append(val)
        finalRead.append(numRow)
    return finalRead


def valueToRGBBins(value):
    if value <= -20.0:
        return (0, 0, 102)
    elif value <= -15.0:
        return (0, 0, 153)
    elif value <= -10.0:
        return (0, 0, 204)
    elif value <= -5.0:
        return (0, 0, 255)
    elif value <= 0.0:
        return (0, 128, 255)
    elif value <= 2.0:
        return (0, 255, 255)
    elif value <= 5.0:
        return (0, 255, 128)
    elif value <= 8.0:
        return (0, 255, 0)
    elif value <= 12.0:
        return (128, 255, 0)
    elif value <= 15.0:
        return (153, 255, 51)
    elif value <= 20.0:
        return (255, 255, 0)
    elif value <= 25.0:
        return (255, 153, 51)
    elif value < 30.0:
        return (255, 128, 0)
    elif value < 35.0:
        return (255, 0, 0)
    elif value < 40.0:
        return (204, 0, 0)
    elif value < 50.0:
        return (153, 0, 0)
    elif value < 60.0:
        return (102, 0, 0)
    else:
        return (0, 0, 0)

    
def makeAndFillImg(data): 
    numRows = len(data)
    numCols = len(data[0])
    img = Image.new('RGB', (numRows, numCols), "blue")
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i,j] = valueToRGBBins(data[i][j])

    return ImageOps.mirror(img.rotate(270, expand=1))

main()



