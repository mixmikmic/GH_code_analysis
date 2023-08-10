# standard setup
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time

# useful helper function
from helpers import imshow

def extractFeaturesFromImage(query_path):
    '''
    @query_path: path of the query image
    @returns: keypoints and descriptors of the query image
    '''
    query_img = cv2.imread(query_path, 0)
    kp_query, des_query = sift.detectAndCompute(query_img, None)  
    return kp_query, des_query, query_img

def initializeMatcher():
    '''
    @returns: FLANN matcher
    '''
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    return flann

def initializeCamera(w):
    '''
    @w: width of the video frame
    @returns: camera object
    '''
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 
    camera.set(cv2.CAP_PROP_EXPOSURE,-4) 
    return camera

def getGoodMatches(des_query, des_scene):
    '''
    @des_query: descriptors of a query image
    @des_scene: descriptors of a scene image
    @returns: list of good matches for query and scene images
    '''
    
    matches = flann.knnMatch(des_query, des_scene,k=2)

    # ratio test as per Lowe's paper
    good_matches = []
    
    # Each member of the matches list must be checked whether two neighbours really exist.
    for m_n in matches:
        if len(m_n) != 2:
            continue
        (m,n) = m_n
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
            
    return good_matches

def findBookSpine(good_matches, query_img, kp_query, kp_scene):
    '''
    @good_matches: set of good matches
    @query_img: query image
    @frame: video frame image
    @kp_query: keypoints of the query image
    @kp_scene: keypoints of the scene image
    @returns: stored bookspine in the correct orientation
    '''
    
    if len(good_matches) > 15:
        
        # Source points and destnation points
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        rows,cols = frame.shape[:2]
        dst = cv2.warpPerspective(query_img, M, (cols, rows));
        return dst
    

# SIFT keypoint extractor
sift = cv2.xfeatures2d.SIFT_create()


kp_query1, des_query1, query_img1 = extractFeaturesFromImage('spine1.jpg')
kp_query2, des_query2, query_img2 = extractFeaturesFromImage('book3.jpg')
kp_query3, des_query3, query_img3 = extractFeaturesFromImage('book2.jpg')

# Initialize FLANN matcher
flann = initializeMatcher()

# Initialize camera
camera = initializeCamera(640)

while True:
    # Get frame at flip it
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # keypoints and descriptors for video frame 
    kp_scene, des_scene =  sift.detectAndCompute(frame_gray,None)
       
    # Good Matches
    good_matches1 = getGoodMatches(des_query1,des_scene)
    good_matches2 = getGoodMatches(des_query2,des_scene)
    good_matches3 = getGoodMatches(des_query3,des_scene)   
        
    dst = findBookSpine(good_matches1, query_img1, kp_query1, kp_scene)

    if type(dst) == np.ndarray: 
        frame = dst
    
    if cv2.waitKey(5) == 32:
        imshow(frame)        
    elif cv2.waitKey(5) == 27:
        break  
   
    cv2.imshow("SIFT Frame", frame)
    
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window

def isolateBookSpineROI(good_matches, query_img, frame, kp_query, kp_scene):
    '''
    @good_matches: set of good matches
    @query_img: query image
    @frame: video frame image
    @kp_query: keypoints of the query image
    @kp_scene: keypoints of the scene image
    
    ASSUMPTION: keypoints on the book spine are distributed uniformly across the height
    '''
    if len(good_matches) > 15:
        
        # Source points and destnation points
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Compute Homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        rows,cols = query_img.shape[:2]
        dst = cv2.warpPerspective(frame, M, (cols, rows));
        
        # Theshold matched points that are below the book spine.
        # If less than 1/5 of all good matches i from the lower part of the book spine,
        # register selection of the book spine.
        threshold_height = int(query_img.shape[0] * 0.7)        
        num_pass_threshold = 0
        for point in src_pts:
            if point[0][1] >= threshold_height:
                   num_pass_threshold += 1
        
        book_selected = False
        threshold_num_matches = int(len(good_matches) / 10)
        
        print "   ", threshold_num_matches, num_pass_threshold        
        selected = True if num_pass_threshold <= threshold_num_matches else False

        return dst, selected
    return None, None

def checkSelected(selected):
    '''
    @selected: book is selected in the current frame
    @returns: True if the book is selected in 10 consecutive frames, and False otherwise
    '''
    global book_selected
    if selected is not None:
        if selected == True:
            book_selected += 1
        else: 
            book_selected = 0

        return book_selected >= 10
    return False

from IPython.display import clear_output

# SIFT keypoint extractor
sift = cv2.xfeatures2d.SIFT_create()


kp_query1, des_query1, query_img1 = extractFeaturesFromImage('spine1.jpg')
kp_query2, des_query2, query_img2 = extractFeaturesFromImage('book3.jpg')
kp_query3, des_query3, query_img3 = extractFeaturesFromImage('book2.jpg')

# Initialize FLANN matcher
flann = initializeMatcher()

# Initialize camera
camera = initializeCamera(640)

global book_selected
book_selected = 0
book_selected_bool = False

while True:
    clear_output(True)

    # Get frame at flip it
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # keypoints and descriptors for video frame 
    kp_scene, des_scene =  sift.detectAndCompute(frame_gray,None)
    
    if book_selected_bool == False:
        # Find Good Matches
        good_matches1 = getGoodMatches(des_query1,des_scene)
        good_matches2 = getGoodMatches(des_query2,des_scene)
        good_matches3 = getGoodMatches(des_query3,des_scene)

        dst, selected = isolateBookSpineROI(good_matches1, query_img1, frame, kp_query1, kp_scene)
        book_selected_bool = checkSelected(selected)
            
    else:
        print "SELECTED"
        
    
    if cv2.waitKey(5) == 32:
        src_pts = np.float32([kp_query1[m.queryIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
        print src_pts      
    elif cv2.waitKey(5) == 27:
        break  
   
    # frame = cv2.flip(frame, 1)
    cv2.imshow("SIFT Frame", frame)
    
cv2.destroyAllWindows()
camera.release()
cv2.waitKey(1) # extra waitKey sometimes needed to close camera window





