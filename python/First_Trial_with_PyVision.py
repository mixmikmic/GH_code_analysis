import pyvision as pv
import pyvision.face.CascadeDetector as cd
import cv2
face_detector = cd.CascadeDetector()
im = pv.Image("sl4.jpg",bw_annotate=True)
#it returns faces collection since theree might be more than one face in the image.
faces = face_detector(im)

for face in faces:
    im.annotatePolygon(face.asPolygon(), width=4)

im.show(delay=0)

'''
cam = pv.Webcam()
while True:
    frame = cam.query()
    rects = detector(frame)
    for rect in rects:
        frame.annotateRect(rect)
    frame.show()
'''
cv2.destroyAllWindows()

import pyvision as pv
import pyvision.face.CascadeDetector as cd
import pyvision.face.FilterEyeLocator as ed
import cv2
face_detect = cd.CascadeDetector()
eye_detect = ed.FilterEyeLocator()
im = pv.Image("sl3.jpg",bw_annotate=True)
faces = face_detect(im)
eyes = eye_detect(im,faces)
for face,eye1,eye2 in eyes:
    im.annotatePolygon(face.asPolygon(), width=4)
    im.annotatePoints([eye1,eye2])

im.show(delay=0)
cv2.destroyAllWindows()



