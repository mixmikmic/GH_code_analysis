# standard setup
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time

# useful helper function
from helpers import imshow

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15), 
                  maxLevel = 3, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

query_path = 'book3.jpg'

# SIFT keypoint detector
sift = cv2.xfeatures2d.SIFT_create()

# Import the query image and compute SIFT keypoints and descriptors
query = cv2.imread(query_path, 0)
kp_query, des_query = sift.detectAndCompute(query, None)  

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Camera settings
camera = cv2.VideoCapture(0)
# reduce frame size to speed it up
w = 640*1.5
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 
camera.set(cv2.CAP_PROP_EXPOSURE,-4) 

KLT = False

# Create some random colors
color = np.random.randint(0,255,(100,3))
i = 0
while True:
    # Get frame at flip it
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    if KLT == True:
        
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        
#         if i == 0:
#             i +=1
#             print "\np1: ", p1
        
        # Select good points
        good_new = p1[status==1]
        good_old = p0[status==1]
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        
        # Compute Homography
#         src_pts = src_pts[status==1]
        M, mask = cv2.findHomography(src_pts, p1, cv2.RANSAC, 1.0)

        # Draw a rectangle that marks the found model in the frame
        h, w = query.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)

        # Draw lines
        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.imshow("SIFT Frame", frame)

        # Now update the previous frame and previous points
        old_gray = gray.copy()
        p0 = good_new.reshape(-1,1,2)
        
    else:
        # Keypoints and descriptors for video frame 
        kp_scene, des_scene =  sift.detectAndCompute(gray,None)

        # Match scene descriptors with query descriptors
        matches = flann.knnMatch(des_query,des_scene,k=2)

        # Ratio test as per Lowe's paper
        good_matches = []
        
        # Each member of the matches list must be checked whether two neighbours really exist.
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        # If enough matches found ...
        if len(good_matches) > 15:

            # Source points and destnation points
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Draw all matched points in the frame
            for pt in dst_pts:
                (x,y) = pt[0]        
                cv2.circle(frame,(int(x),int(y)), 2, (0,255,0), 10)
            
        cv2.imshow("SIFT Frame", frame)
    

    if cv2.waitKey(5) == 32:
        p0 = dst_pts     
        print p0
        KLT = True
        old_frame = frame
        old_gray = gray
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
    elif cv2.waitKey(5) == 27:
        break  
   
    
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window





query_path = 'book5.jpg'

# SIFT keypoint detector
sift = cv2.xfeatures2d.SIFT_create()

# Import the query image and compute SIFT keypoints and descriptors
query = cv2.imread(query_path, 0)
kp_query, des_query = sift.detectAndCompute(query, None)  

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Camera settings
camera = cv2.VideoCapture(0)
# reduce frame size to speed it up
w = 640*1.5
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 
camera.set(cv2.CAP_PROP_EXPOSURE,-4) 


# Create some random colors
color = np.random.randint(0,255,(100,3))

iteration = 0
i = 0
while True:
    # Get frame and flip it
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    if iteration % 15 != 0:
        if len(p0) == 0 or len(src_pts) == 0:
            continue
            
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        
        
        print len(src_pts), len(p1), len(p0), len(status)

        # Select good points
        good_new = p1[status==1]
        good_old = p0[status==1]
        good_src = src_pts[status==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                
        print "Length source = {}, length destination = {}".format(len(src_pts), len(p1))
        
        # Compute Homography
        M, mask = cv2.findHomography(src_pts, p1, cv2.RANSAC, 5.0)

        # Draw a rectangle that marks the found model in the frame
        h, w = query.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)

        # Draw lines
        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.imshow("SIFT Frame", frame)

        # Now update the previous frame and previous points
        old_gray = gray.copy()
        p0 = good_new.reshape(-1,1,2)
        src_pts = good_src.reshape(-1, 1, 2)
        
        iteration+=1
        
    else:
        
        # Keypoints and descriptors for video frame 
        kp_scene, des_scene =  sift.detectAndCompute(gray,None)

        # Match scene descriptors with query descriptors
        matches = flann.knnMatch(des_query,des_scene,k=2)

        # Ratio test as per Lowe's paper
        good_matches = []
        
        # Each member of the matches list must be checked whether two neighbours really exist.
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        # If enough matches found ...
        if len(good_matches) > 15:

            # Source points and destnation points
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            
            # Draw all matched points in the frame
            for pt in dst_pts:
                (x,y) = pt[0]        
                cv2.circle(frame,(int(x),int(y)), 2, (0,255,0), 10)
            
            
        
            # Compute Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Draw a rectangle that marks the found model in the frame
            h, w = query.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, M)

            # Draw lines
            cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA) 
        
               
            
            # save previous points
            p0 = dst_pts     

            # save old frame
            old_frame = frame
            old_gray = gray
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
            iteration+=1

    
        cv2.imshow("SIFT Frame", frame)
    
    if cv2.waitKey(5) == 27:
        break  
   
    
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window

query_path = 'book5.jpg'

# SIFT keypoint detector
sift = cv2.xfeatures2d.SIFT_create()

# Import the query image and compute SIFT keypoints and descriptors
query = cv2.imread(query_path, 0)
kp_query, des_query = sift.detectAndCompute(query, None)  

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Camera settings
camera = cv2.VideoCapture(0)
# reduce frame size to speed it up
w = 640*1.5
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 
camera.set(cv2.CAP_PROP_EXPOSURE,-4) 


# Create some random colors
color = np.random.randint(0,255,(100,3))

iteration = 0
while True:
    # Get frame and flip it
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    if iteration % 15 != 0:
        if len(p0) == 0 or len(src_pts) == 0:
            continue
            
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        
        # Select good points
        good_new = p1[status==1]
        good_old = p0[status==1]
        good_src = src_pts[status==1]

#         # draw the tracks
#         for i,(new,old) in enumerate(zip(good_new,good_old)):
#             a,b = new.ravel()
#             c,d = old.ravel()
#             mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#             frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                    
        # Compute Homography
        M, mask = cv2.findHomography(src_pts, p1, cv2.RANSAC, 5.0)

        # Draw a rectangle that marks the found model in the frame
        h, w = query.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)

        # Draw lines
        cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.imshow("SIFT Frame", frame)

        # Now update the previous frame and previous points
        old_gray = gray.copy()
        p0 = good_new.reshape(-1,1,2)
        src_pts = good_src.reshape(-1, 1, 2)
        
        iteration+=1
        
    else:
        
        # Keypoints and descriptors for video frame 
        kp_scene, des_scene =  sift.detectAndCompute(gray,None)

        # Match scene descriptors with query descriptors
        matches = flann.knnMatch(des_query,des_scene,k=2)

        # Ratio test as per Lowe's paper
        good_matches = []
        
        # Each member of the matches list must be checked whether two neighbours really exist.
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m,n) = m_n
            if m.distance < 0.7*n.distance:
                good_matches.append(m)

        # If enough matches found ...
        if len(good_matches) > 15:

            # Source points and destnation points
            src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
          
#             # Draw all matched points in the frame
#             for pt in dst_pts:
#                 (x,y) = pt[0]        
#                 cv2.circle(frame,(int(x),int(y)), 2, (0,255,0), 10)
        
            # Compute Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Draw a rectangle that marks the found model in the frame
            h, w = query.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, M)

            # Draw lines
            cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA)            
            
            # save previous points
            p0 = dst_pts     

            # save old frame
            old_frame = frame
            old_gray = gray
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)
            iteration+=1

    
        cv2.imshow("SIFT Frame", frame)
    
    if cv2.waitKey(5) == 27:
        break  
   
    
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window





