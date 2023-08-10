import string
def changeFileSize(x):
    tempstring = str(x)
    tempstring = tempstring.replace(",","")
    if str(x).endswith("k"):
        tempstring = tempstring.rstrip("k") 
        tempfloat = float(tempstring)
        return float(tempfloat/1000)
    if str(x).endswith("M"):
        tempstring = tempstring.rstrip("M")
        return float(tempstring)
    else:
        return int(0.0)

