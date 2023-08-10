from IPython.display import Image as Img
Img(filename="/project/2017/pawel/heatmap_Canada_allpages_1860-1869.png")

Img(filename="/project/2017/pawel/heatmap_Canada_page1only_1860_1869.png")

from IPython.display import Image as Img

Img(filename="/project/2017/pawel/sample_text.png")

get_ipython().system(' tesseract /project/2017/pawel/sample_text.png stdout')

import pytesseract

get_ipython().system(' pip install pytesseract           ')

import pytesseract
from PIL import Image
im=Image.open('/project/2017/pawel/sample_text.png')
resultOCR=pytesseract.image_to_string(im)
print(resultOCR)

tmpfile='view_image.png'
im = Image.open('/project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29-006.tif')
im.save(tmpfile)
Img(filename=tmpfile)

get_ipython().system(' grep Somme /project/2017/pawel/output_0FFO-1916-SEP29-006.txt')

get_ipython().system(' grep Somme /project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29.xml')

from bs4 import BeautifulSoup

def process_file(f):
    
    output_list=[]
    soup = BeautifulSoup(f,"lxml")
    taglist = soup.find_all("text")
    
    for tagtxt in taglist:
        termfound=False
        pageinfo=tagtxt.find_all("pg")
        words = tagtxt.find_all("wd")
        for w in words:
            if(w.string == searchterm):
                termfound=True
                #print("Found")
                
        if(termfound==True):
            for p in pageinfo:
                textbox=p["pos"].split(",")
                textbox=map(int,textbox)
                pagenumber=p["pgref"]
                imagefile=f.name.strip(".xml")+"-"+pagenumber.zfill(3)+".tif"
                #print(imagefile)
                im = Image.open(imagefile)
                im_cropped = im.crop(textbox)
                im_cropped.save("image_for_ocr.png")
                textOCR = pytesseract.image_to_string(im_cropped)
                #print(textOCR)
                print("number found by tesseract",textOCR.count(searchterm))
                output_list.append([imagefile,textbox])
                
    return output_list

searchterm="Somme"
f=open("/project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29.xml",encoding="iso-8859-1")
outputlist=process_file(f)
f.close()

Img(filename='image_for_ocr.png')

from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

def process_file(f):
    
    output_list=[]
    
    soup = BeautifulSoup(f,"lxml")
    taglist = soup.find_all("text")
    
    for tagtxt in taglist: 
        pageinfo=tagtxt.find_all("pg")
        
# check if search term occurs in text tag
        termfound=False
        words=tagtxt.find_all("wd")
        for w in words:
            if(w.string == searchterm):
                termfound=True
                print("FOUND")

        if(termfound==True):
            for p in pageinfo:
                textbox=p["pos"].split(",")
                textbox=map(int,textbox) # convert 4 strings to 4 integers
                pagenumber=p["pgref"]
                imagefile=f.name.strip(".xml")+"-"+pagenumber.zfill(3)+".tif"
                output_list.append([imagefile,textbox ])
            
    return output_list

def process_image_file(fim):
    
    list_of_boxes=[]
    for element in output_list:
        if element[0]==fim.name:
            list_of_boxes.append(element[1])
    
    totalfound=0
    if( len(list_of_boxes)>0 ):
        
        im = Image.open(fim)        
        for box in list_of_boxes:
            im_cropped = im.crop(box)
            textOCR = pytesseract.image_to_string(im_cropped)
            totalfound=totalfound+textOCR.count("Somme")
            
    return totalfound
            
searchterm = "Somme"
f=open("/project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29.xml",encoding="iso-8859-1")
output_list=process_file(f)
f.close()
print(output_list)


file_image=open("/project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29-006.tif","rb")
result=process_image_file(file_image)
file_image.close()

print("total found is ",result)

import pyspark
sc = pyspark.SparkContext()

from bs4 import BeautifulSoup
from PIL import Image
import io
import pytesseract

def process_file_spark(something):
    
    filename=something[0]
    f=something[1]
    output_list=[]
    
    soup = BeautifulSoup(f,"lxml")
    taglist = soup.find_all("text")
    
    for tagtxt in taglist: 
        pageinfo=tagtxt.find_all("pg")
        
# check if search term occurs in text tag
        termfound=False
        words=tagtxt.find_all("wd")
        for w in words:
            if(w.string == searchterm):
                termfound=True
                print("FOUND")

        if(termfound==True):
            for p in pageinfo:
                textbox=p["pos"].split(",")
                textbox=map(int,textbox) # convert 4 strings to 4 integers
                pagenumber=p["pgref"]
                imagefile=filename.strip(".xml")+"-"+pagenumber.zfill(3)+".tif"
                output_list.append([imagefile,textbox ])
            
    return output_list

def process_image_file_spark(something):
    
    filename=something[0]
    list_of_boxes=[]
    for element in output_list:
        for element1 in element:
            if element1[0]==filename:
                list_of_boxes.append(element1[1])
    
    totalfound=0

    if( len(list_of_boxes)>0 ):
        image_data=io.BytesIO(something[1])
        im=Image.open(image_data)
        
        for box in list_of_boxes:
            im_cropped = im.crop(box)
            textOCR = pytesseract.image_to_string(im_cropped)
            totalfound=totalfound+textOCR.count(searchterm)
    return filename,totalfound

import time
t0 = time.time()

searchterm = "Somme"

#xmlfiles=sc.wholeTextFiles('/project/datasets/PageScans/1916/19160929/0FFO-1916-SEP29.xml')
xmlfiles=sc.wholeTextFiles('/project/datasets/PageScans/1916/1916092[0-9]/*.xml')

xmlfiles.cache()
output=xmlfiles.map(process_file_spark)
output.cache()
output_list=output.collect()

files=output.flatMap(lambda el:el).map(lambda el:el[0]).distinct().collect()
    
#imgfiles=sc.binaryFiles(','.join(files),800)
imgfiles=sc.binaryFiles('/project/datasets/PageScans/1916/1916[0-9][0-9][0-9][0-9]/*.tif')         .filter(lambda el:el[0] in files)

print("getNumPartitions",imgfiles.getNumPartitions())
print(imgfiles.count())

globalcount = imgfiles.map(process_image_file_spark)
print("count is ",globalcount.collect())
t1 = time.time()
print("total time was ",t1-t0)

sc.stop()

conf = pyspark.SparkConf().set("spark.executor.memory", "13g")
sc = pyspark.SparkContext(master='spark://compute7.paul:7077', conf=conf)

pwd



