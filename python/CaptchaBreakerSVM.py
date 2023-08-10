# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:42:09 2017

@author: roliman
"""


#src = elem_c.get_attribute('src')
#save image
#img = img.convert('RGBA')


#out = im.filter(ImageFilter.SHARPEN)
#out = out.convert("L")
#img = out.filter(ImageFilter.SMOOTH_MORE)
#C:\Program Files (x86)\Tesseract-OCR\
#pixdata = im.load()
#or y in range(im.size[1]):
  #     for x in range(im.size[0]):
   #        if pixdata[x, y][0] < 100:
               # make dark color black
    #           pixdata[x, y] = (255, 255, 255, 255)
     #      else:
               # make light color white
      #         pixdata[x, y] = (0, 0, 0, 255)              

#from skimage.io import imread
#im = imread("farm.jpg")


# Import datasets, classifiers and performance metrics


#do something similar for the labels

# Create a classifier: a support vector classifier

#n_samples = len(Xlist)
#Xlist = np.hstack(Xlist)
#Ylist = np.hstack(Ylist)
#data = Xlist.reshape((n_samples, -1))

#clf=AdaBoostClassifier(n_estimators=100)
#scores = cross_val_score(clf, Xlist, Ylist)
#print(scores.mean())


#binarization
#line removal
#discontinuity removal
#dot removal
#segmentation
#feature extraction

#Classifying Features (raw pixels) libsvm
# training:
#   data: integer arrays
#   target: arrays of 0-35(represents[0-9A-Z])
#   clf.fit(data, target)

# predicting:
#   array = preprocess char image into arrays
#   code = clf.predict(array)
#   char = lookup code in [0-9A-Z]


#use txt
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

#input result
#txt = 'b8uQM'


#while page charateristic = x
   #repeat operation
   #elem_x = driver.find_element_by_xpath('//*[(@id = "id_submit")]')
   #elem_x.click()
   
##############################################################################

#receita tipo 1 e tipo 2 e NIITR

import pandas as pd 
import requests 
from bs4 import BeautifulSoup
from scipy import ndimage
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pytesseract
from PIL import Image
import scipy.ndimage
from sklearn import datasets, svm, metrics
import os
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
from scipy import misc # feel free to use another image loader
import glob
import matplotlib.pyplot as plt
from selenium.webdriver.common.action_chains import ActionChains

urls = ['http://www.receita.fazenda.gov.br/Aplicacoes/ATSPO/Certidao/CndConjuntaInter/InformaNICertidao.asp?Tipo=2', 
        'http://www.receita.fazenda.gov.br/Aplicacoes/ATSPO/Certidao/CndConjuntaInter/InformaNICertidao.asp?Tipo=1', 
        'http://www.receita.fazenda.gov.br/Aplicacoes/ATSPO/Certidao/CertInter/NIITR.asp']

cpfs= [ '54615593953',
        '76999637920',
        '97664049920',
        '52518329900',
        '75902486904',
        '25971607855',
        '12747278808',
        '27171406890', 
        '73821918691',
        '07323516773',
        '04454948674',
        '04410902814',
        '06933024458',
        '40806022434',
        '29272629500', 
        '20823410625',
        '59013435653',
        '82453519891',
        '02738821804',
        '02547063808',
        '07380101897',
        '13986856404',
        '78973554115',
        '26450542691']

#cpfs
#cpfs = pd.read_csv('C:\Crawlers Enforce\cpfs.csv', header = None, encoding = 'utf-8')
#cpfsA =  cpfs['CPF/CNPJ']
#cpfs = cpfs.dropna()
import numpy
path = 'C:/Crawlers Enforce/Train Data/'  

Xlist = []
Ylist = []
for directory in os.listdir(path):
  #  print(directory)
    for img in os.listdir(path+directory):
        img = Image.open(path+directory+'/'+img)
        fvector = numpy.array(img).flatten()[:50]#.flatten()
    
        Xlist.append(fvector)
        Ylist.append(directory)
print(len(Xlist))
    

classifier = svm.SVC(kernel='rbf', C=128, gamma=0.01)

classifier.fit(Xlist, Ylist)
predicted = classifier.predict(Xlist)
expected = Ylist
print(metrics.classification_report(expected, predicted))


for url in urls: 
    for i in cpfs:
        with open('C:\Crawlers Enforce\CAR\diretorios\cpf{}.xls'.format(i[:3]), 'wb') as output: 
      
        #get captcha image 
        #driver = webdriver.Ie("P:\CAR\IEDriverServer.exe")
        
            driver = webdriver.PhantomJS("P:\CAR\phantomjs.exe")
            driver.set_window_size(1024, 768)
            driver.get(url)
            elem_c = driver.find_element_by_xpath('//*[(@id = "imgCaptchaSerpro")]')
            ActionChains(driver).move_to_element(elem_c).perform()
        
            location = elem_c.location
            size = elem_c.size
            driver.save_screenshot('receita_shot.png')

            im = Image.open('receita_shot.png')
            left = location['x']
            top = location['y']
            right = location['x'] + size['width']
            bottom = location['y'] + size['height']

            im = im.crop((left, top, right, bottom)) # defines crop points
            im.save('receita_shot.png')
            im
        
            im2 = ndimage.binary_dilation(im)
            scipy.misc.imsave('dilation3.png', im2)
            im = Image.open('dilation3.png')
        
      #  try:
            big = im.resize((330, 70), Image.NEAREST)
            big.save('BWT.tiff', dpi=(300,300) )
            
            pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
            txt = pytesseract.image_to_string(im)
            txtb = pytesseract.image_to_string(big)
            print(txt, txtb)
                
     #   else:
            box1 = (0,0,32,180) 
            seg1 = im.crop(box1)
            seg1.save('box670.png')

            box2 = (28,0,60,180)
            seg2 = im.crop(box2)
            seg2.save('box671.png')

            box3 = (60,0,92,180)
            seg3 = im.crop(box3)
            seg3.save('box672.png')

            box4 = (90,0,122,180)
            seg4 = im.crop(box4)    
            seg4.save('box673.png')
    
            box5 = (116,0,148,180)
            seg5 = im.crop(box5)
            seg5.save('box674.png')

            box6 = (148,0,180,180)
            seg6 = im.crop(box6)
            seg6.save('box675.png')
           
        
            import numpy 
            def decode_captcha(img1, img2, img3, img4, img5, img6, func=classifier.predict):
                x1 = numpy.array(img1).flatten()[:50] #binary colors input?
                x2 = numpy.array(img2).flatten()[:50]
                x3 = numpy.array(img3).flatten()[:50]
                x4 = numpy.array(img4).flatten()[:50]
                x5 = numpy.array(img5).flatten()[:50]
                x6 = numpy.array(img6).flatten()[:50]
                func =  classifier.predict
                txt = str(str(func(x1))[2:-2]+str(func(x2))[2:-2]+str(func(x3))[2:-2]+str(func(x4))[2:-2]+str(func(x5))[2:-2]+str(func(x6))[2:-2])
                #find 'and replace by empty space
                return txt
        
            get = decode_captcha(seg1, seg2, seg3, seg4, seg5, seg6)
            print(get)
        
     
            body = {'NI' : i, # '26450542691',
                    'Tipo' : '2', #url[:]
                   'txtTexto_captcha_serpro_gov_br' : txt }
    
#
            response = requests.post(url, data = body) 
            output.write(response.content)
        


##############################################################################

#receita consulta Cafir (NIRF)
#
#url = 'https://coletorcafir.receita.fazenda.gov.br/coletor/consulta/consultaCafir.jsf'
#
#for i in cpfs:
#    with open('C:\Crawlers Enforce\CAR\diretorios\cpf{}.xls'.format(i[:3]), 'wb') as output: 
#
#        body = {'form' : 'form',
#                'form:nirfImovel':'2222222-2'
#                'form:j_idt24:textoCaptcha' : txt, 
#                'javaz.faces.ViewState' : '-4077935195633330508:-2716986253673257627'} #weird thingy
#
###############################################################################
#
##receita consulta CPF
#
#url = 'https://www.receita.fazenda.gov.br/Aplicacoes/SSL/ATCTA/CPF/ConsultaSituacao/ConsultaPublica.asp' 
#bday = '08071994'
#
#for i in cpfs:
#    with open('C:\Crawlers Enforce\CAR\diretorios\cpf{}.xls'.format(i[:3]), 'wb') as output: 
#        
#        driver = webdriver.Ie("P:\CAR\IEDriverServer.exe")
#        #driver = webdriver.PhantomJS("P:\CAR\phantomjs.exe")
#        #driver.set_window_size(1024, 768)
#        driver.get(url)
#
#        elem3 =  driver.find_element_by_xpath('//*[(@id = "txtCPF")]')
#        elem3.click()
#        elem3.send_keys(i)
#
#        elem4 = driver.find_element_by_xpath('//*[(@id = "txtDataNascimento")]')
#        elem4.click()
#        elem4.send_keys('08071994')
#
#
#        elem_x = driver.find_element_by_xpath('//*[(@id = "id_submit")]')
#        elem_x.click()
#
#        elem_c = driver.find_element_by_xpath('//*[(@id = "imgCaptcha")]')
#
#        location = elem_c.location
#        size = elem_c.size
#        driver.save_screenshot('receita_shot.png')
#
#        im = Image.open('receita_shot.png')
#        left = location['x']
#        top = location['y']
#        right = location['x'] + size['width']
#        bottom = location['y'] + size['height']
#
#        im = im.crop((left, top, right, bottom)) # defines crop points
#        im.save('receita_shot.png')
#        im
#        
#        im2 = ndimage.binary_dilation(im)
#        scipy.misc.imsave('dilation3.png', im2)
#        im = Image.open('dilation3.png')
#        
#        big = im.resize((330, 70), Image.NEAREST)
#        big.save('BWT.tiff', dpi=(300,300) )
#            
#        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
#        txt = pytesseract.image_to_string(im)
#        txtb = pytesseract.image_to_string(big)
#        print(txt, txtb)
#        
#        seg1 = im.crop(box1)
#        seg1.save('box670.png')
#
#        box2 = (28,0,60,180)
#        seg2 = im.crop(box2)
#        seg2.save('box671.png')
#
#        box3 = (60,0,92,180)
#        seg3 = im.crop(box3)
#        seg3.save('box672.png')
#
#        box4 = (90,0,122,180)
#        seg4 = im.crop(box4)    
#        seg4.save('box673.png')
#    
#        box5 = (116,0,148,180)
#        seg5 = im.crop(box5)
#        seg5.save('box674.png')
#
#        box6 = (148,0,180,180)
#        seg6 = im.crop(box6)
#        seg6.save('box675.png')
#        
#        path = 'C:/Crawlers Enforce/Train Data/'  
#        
#        import numpy 
#        def decode_captcha(img1, img2, img3, img4, img5, img6, func=classifier.predict):
#                x1 = numpy.array(img1).flatten()[:50]
#                x2 = numpy.array(img2).flatten()[:50]
#                x3 = numpy.array(img3).flatten()[:50]
#                x4 = numpy.array(img4).flatten()[:50]
#                x5 = numpy.array(img5).flatten()[:50]
#                x6 = numpy.array(img6).flatten()[:50]
#                func =  classifier.predict
#                txt = str(str(func(x1))[2:-2]+str(func(x2))[2:-2]+str(func(x3))[2:-2]+str(func(x4))[2:-2]+str(func(x5))[2:-2]+str(func(x6))[2:-2])
#                #find 'and replace by empty space
#                return txt
#        
#        get = decode_captcha(seg1, seg2, seg3, seg4, seg5, seg6)
#        print(get)
#        
#        elem_c1 = driver.find_element_by_xpath('//*[(@id = "txtTexto_captcha_serpro_gov_br")]')
#        elem_c1.send_keys(txt)
#
#        #click baixar dados 
#        elem6 = driver.find_element_by_xpath('//*[(@id = "btnAR2")]')
#        elem6.click() 
#        
#        body = {'txtTexto_captcha_sepro_gov_br' : txt, # '26450542691',
#                'tempTxtCPF' : i, #url[:]
#                'tempTxtNascimento': bday
#                'temptxtTexto_captcha_serpro_gov_br' : txt }
#        
#        
#        response = requests.post(url, data = body) 
#        output.write(response.content)


