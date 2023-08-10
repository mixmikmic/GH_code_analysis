#Installing dependencies
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#reading images and convertion them to grayscale
img1 = cv.imread('bot.jpg')
img3 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('p_area.jpg')
img4 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

#instance of ORB
orb = cv.ORB()

#keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#finding matches
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

#top 40 matches
matches = matches[:40]

# finding angle
if matches[0]!=None:
    src_points = np.float32(map(lambda x: kp1[x.queryIdx].pt, matches[:40])).reshape(-1,1,2)
    dst_points = np.float32(map(lambda x: kp2[x.trainIdx].pt, matches[:40])).reshape(-1,1,2)
    
    H, _ = cv.findHomography(src_points, dst_points)
    
    p1 = H.dot([1,1,1])
    p2 = H.dot([2,2,1])
    
    p1 = p1 / p1[-1]
    p2 = p2 / p2[-1]
    
    org_line_seg = np.array([2,2]) - np.array([1,1])
    new_line_seg = p2[:2] - p1[:2]
    
    angle = np.dot(org_line_seg, new_line_seg) / np.sqrt(np.sum((org_line_seg ** 2)) * np.sum(new_line_seg ** 2))
    
    theta = np.array([math.asin(angle), math.acos(angle)])
    degree = theta*180*7/22
    print degree
	
#drawing matches
def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        
    cv.imshow('Matched Features', out)
    cv.waitKey(0)
    cv.destroyWindow('Matched Features')
    
    return out

	
out = drawMatches(img3, kp1, img4, kp2, matches[:40])



