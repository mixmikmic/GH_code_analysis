import sys
sys.path.append('../')
import python.edge as edge 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import cv2
def s(img):
    cv2.imshow("a",img)
    cv2.waitKey(0)

file= '../data/interim/s7_spim3/361.jpg'
img = cv2.imread(file,0)
app1 = cv2.imread(file)
app2 = app1.copy()
app3 = app1.copy()
# cv2.imshow("a",img)
# cv2.waitKey(0)
filtered = cv2.bilateralFilter(img, 12,1050,1050)
edgeShape = edge.edge(filtered)
plt.imshow(cv2.cvtColor(edgeShape, cv2.COLOR_GRAY2RGB))

k, thres = cv2.threshold(edgeShape, 1, 255, cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(thres, cv2.COLOR_GRAY2RGB))

opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),iterations=1)
plt.imshow(cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB))

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),iterations=2)
plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB))

erode = cv2.erode(closing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
plt.imshow(cv2.cvtColor(erode, cv2.COLOR_GRAY2RGB))

_ , contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(app1, contours, -1, (250, 0, 0),2)
plt.imshow(app1)
s(app1)

k, thres = cv2.threshold(edgeShape, 2, 255, cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(thres, cv2.COLOR_GRAY2RGB))

closing = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)),iterations=1)
plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB))

opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),iterations=1)
plt.imshow(cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB))

_ , contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(app2, contours, -1, (250, 0, 0),2)
plt.imshow(app2)
s(app2)

cannyEdge = edge.canny(filtered)
plt.imshow(cv2.cvtColor(cannyEdge, cv2.COLOR_GRAY2RGB))

dilate = cv2.dilate(cannyEdge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_GRAY2RGB))

closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),iterations=1)
plt.imshow(cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB))

_ , contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(app3, contours, -1, (250, 0, 0),2)
plt.imshow(app3)









