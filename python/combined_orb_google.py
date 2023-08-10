###########################################
# Google doc and ORB setup (run once)
###########################################

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)
 
# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
sheet = client.open("Xilinx_Hackathon_2017").sheet1


###### ORB
# imports for ORB
import numpy as np
import cv2

# Initiate ORB detector
orb = cv2.ORB_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

import string
import re
####################################
######   Configurable items:  ######
####################################
# arbitary number. Fine tune once you get here
thres_hold = 250

#TODO: receive the cropped face from the webcam

# New face from the webcam
#TODO update this variable to reference the webcam image
image = '1.JPG'

# Find the Key point
img = cv2.imread(image,0)          # queryImage

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img,None)
#kp2, des2 = orb.detectAndCompute(img2,None)

###########################################
# Read from the Google doc
###########################################

# Extract and print all of the values
all_faces_str = sheet.get_all_values()
print(all_faces_str)



# convert google doc strings to numbers

data = all_faces_str[1:]
face_order = 0
for row in data:
    
    face_order += 1
    myTotal = np.array([])
    for element in row:
        myArray = np.array(eval(element))
        myTotal = np.vstack([myTotal,myArray]) if myTotal.size else myArray

        
    # convert data type
    myTotal_uint8 = np.uint8(myTotal) 
    
    # Add each image descriptor list from the database
    clusters = np.array([myTotal_uint8])
    bf.add(clusters)
    
    # Train: Does nothing for BruteForceMatcher though.
    bf.train()
    
    matches = bf.match(des1,myTotal_uint8)
    matches = sorted(matches, key = lambda x:x.distance)
    
    facest_face = 0
    numb_matches = (len(matches))
    if numb_matches > facest_face:
        facest_face = numb_matches
    
    face_found = False
    if facest_face > thres_hold:
        face_found = True
        found_face_order = face_order

    print(facest_face)
    print(face_found)

# Add face to database if not found, otherwise report the face

if face_found == True:
    print ("FACE FOUND! Face number:")
    print (found_face_order)
else:
    print ("Face NOT found. adding face to database")
    sheet.append_row('')
    col_in = 1
    row_in = sheet.row_count

    collumn_list = []
    des_cnt = 0
    des_per_cell = 100
    for collumn in des1:

        collumn_list.append(collumn.tolist())
        # Append des_per_cell # of collumns into a single collumn and upload
        if des_cnt >= des_per_cell -1:
            print ("cloud")
            sheet.update_cell(row_in, col_in, collumn_list)
            collumn_list = []
            col_in += 1
            des_cnt = 0
        else:
            des_cnt += 1



# Calculate the results
for tests in matches:
    numb_matches = (len(matches))
    if facest_face > numb_matches:
        facest_face = numb_matches


    if facest_face > thres_hold:
        face_found = True

#row_float = list(map(float,row))
#all_faces_flt.append(row_float)

facest_face
print(facest_face)
print(face_found)
#print(all_faces_flt)

# Final result from the face detection characterizaion list is
# all_faces_flt
# Use where needed

###########################################
# Write to the Google doc
###########################################

# dumby data converted to text. Set "face_data" to the row you wish to append
face_data = [4.5, 6, 8, 2.133, 35623, 8]

# convert the float array to a string array before adding to the google doc
row = list(map(str, face_data))
print(row) 

