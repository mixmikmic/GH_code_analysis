# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

import dlib
import cv2
import numpy as np

#Classifier constants
RIGHT_EYE_POINTS = range(36, 42)
LEFT_EYE_POINTS = range(42, 48)

LEFT_BROW_POINTS = range(22, 27)
RIGHT_BROW_POINTS = range(17, 22)
NOSE_POINTS = range(27, 35)
MOUTH_POINTS = range(48, 61)

POINTS = [RIGHT_EYE_POINTS, LEFT_EYE_POINTS, LEFT_BROW_POINTS, RIGHT_BROW_POINTS, NOSE_POINTS, MOUTH_POINTS]
FLAT_POINTS = range(68)
#alternatively flatten the other list with [item for sublist in POINTS for item in sublist]

#paths of helper files
PRED_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_PATH = "haarcascade_frontalface_default.xml"
IMG_PATH = "house.jpg" #trump.jpg

#constants for later use
SCALE_FACTOR = 0.75 #smaller = faster but blurrier
FEATHER_AMOUNT = int(25*SCALE_FACTOR)//2*2+1 #adapt to scaling-factor. assuming 640x480
COLOUR_CORRECT_BLUR_FRAC = 0.6 #ratio of overlay of blur

#configure if get_landmarks should use the dlib-detector or the cascade classifier
DLIB_ON = True
USE_COLOR_CORRECTION = True
USE_MASK = True

#initializations
cascade = cv2.CascadeClassifier(HAAR_PATH)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)


def get_landmarks(im):
    rects = detector(im, 1) if DLIB_ON else cascade.detectMultiScale(im, 1.3,5)
    
    if len(rects) != 1:
        raise RuntimeError("Nr. of faces: "+str(len(rects)))
    
    if DLIB_ON:
        rect = rects[0]
    else:
        x,y,w,h = rects[0]
        rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h)) 
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in POINTS:
        points = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, points, color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    
    
def transformation_from_points(points1, points2):
    """ Solve the procrustes problem by subtracting centroids, scaling by the
    standard deviation, and then using the SVD to calculate the rotation. See
    the following for more details:
    https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem """
    
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    s1 = np.std(points1)
    s2 = np.std(points2)

    points1 = (np.array(points1)-c1)/s1
    points2 = (np.array(points2)-c2)/s2
    
    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([  np.hstack(((s2 / s1) * R,
                        c2.T - (s2 / s1) * R * c1.T)),
                        np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                              np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)//2*2+1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))


def face_swap(im1, im2, landmarks2, mask2, color_correct=True, masking=True):
    """Takes precomputed values for the image to insert as parameters"""
    #downscale webcam for performance
    im1 = cv2.resize(im1, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    
    #get landmarks for webcam
    landmarks1 = get_landmarks(im1)
    
    #calculate the transformation matrix that shifts + stretches the overlay-image
    M = transformation_from_points(landmarks1[FLAT_POINTS], landmarks2[FLAT_POINTS])
    warped_mask = warp_im(mask2, M, im1.shape)
    warped_im2 = warp_im(im2, M, im1.shape)
    
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask],axis=0)
    
    if color_correct:
        warped_im2 = correct_colours(im1, warped_im2, landmarks1)

    if masking:
        output_im = im1 * (1.0 - combined_mask) + warped_im2 * combined_mask
    else:
        output_im = warped_im2
    
    #convert image to displayable format
    _, im_arr = cv2.imencode('.jpg', output_im)
    image = cv2.imdecode(im_arr, cv2.IMREAD_COLOR)    
    frame = cv2.resize(image,None,fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation = cv2.INTER_LINEAR)
    
    return frame


#actual script
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    #precompute data for 2nd image once
    im2 = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    im2landmarks = get_landmarks(im2)
    mask2 = get_face_mask(im2, im2landmarks)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        try:
            frame = face_swap(frame, im2, im2landmarks, mask2, USE_COLOR_CORRECTION, USE_MASK)
        except RuntimeError as e: #Print error into image
            frame = cv2.putText(frame, str(e), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow("Face", frame)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

