import os
import cv2
import prettywebsite
import matplotlib.pyplot as plt
import pkg_resources

print("PrettyWebsite:",prettywebsite.__version__)

img1 = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample.png'))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample2.png'))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample3.png'))
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

img1_b = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample.png'),0)
img2_b = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample2.png'),0)
img3_b = cv2.imread(pkg_resources.resource_filename('prettywebsite','../share/data/sample3.png'),0)

plt.figure("Images",figsize=(12, 6), dpi=80)
plt.subplot(1,3,1)
plt.title("Sample 1")
plt.imshow(img1)
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(1,3,2)
plt.title("Sample 2")
plt.imshow(img2)
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(1,3,3)
plt.title("Sample 3")
plt.imshow(img3)
plt.xticks([],[])
plt.yticks([],[])

VC_1_W =  os.stat(pkg_resources.resource_filename('prettywebsite','../share/data/sample.png')).st_size
VC_2_W =  os.stat(pkg_resources.resource_filename('prettywebsite','../share/data/sample2.png')).st_size
VC_3_W =  os.stat(pkg_resources.resource_filename('prettywebsite','../share/data/sample3.png')).st_size

VC_1_QT = len(prettywebsite.quadTreeDecomposition.quadTree(img1_b,minStd=5,minSize=10).blocks)
VC_2_QT = len(prettywebsite.quadTreeDecomposition.quadTree(img2_b,minStd=5,minSize=10).blocks)
VC_3_QT = len(prettywebsite.quadTreeDecomposition.quadTree(img3_b,minStd=5,minSize=10).blocks)

plt.figure("W vs QT")
plt.title("Visual complexity by weight vs Visual complexity by QuadTree")
plt.xlabel("Weight")
plt.ylabel("QuadTree")
plt.scatter(VC_1_W, VC_1_QT)
plt.scatter(VC_2_W, VC_2_QT)
plt.scatter(VC_3_W, VC_3_QT)
plt.show()

S_1 = prettywebsite.symmetry.getSymmetry(img1_b,5,20)
S_2 = prettywebsite.symmetry.getSymmetry(img2_b,5,20)
S_3 = prettywebsite.symmetry.getSymmetry(img3_b,5,20)

plt.figure("Symmetry",figsize=(12, 3), dpi=80)
plt.suptitle("Degree of Symmetry")
plt.subplot(1,3,1)
plt.title(S_1)
plt.imshow(img1)
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(1,3,2)
plt.title(S_2)
plt.imshow(img2)
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(1,3,3)
plt.title(S_3)
plt.imshow(img3)
plt.xticks([],[])
plt.yticks([],[])
plt.show()

C_1_S = prettywebsite.colorfulness.colorfulnessHSV(img1)
C_2_S = prettywebsite.colorfulness.colorfulnessHSV(img2)
C_3_S = prettywebsite.colorfulness.colorfulnessHSV(img3)

C_1_RGB = prettywebsite.colorfulness.colorfulnessRGB(img1)
C_2_RGB = prettywebsite.colorfulness.colorfulnessRGB(img2)
C_3_RGB = prettywebsite.colorfulness.colorfulnessRGB(img3)

plt.figure("S vs RGB")
plt.title("Colorfulness estimation in HSV and RGB color spaces")
plt.xlabel("S")
plt.ylabel("RGB")
plt.scatter(C_1_S, C_1_RGB)
plt.scatter(C_2_S, C_2_RGB)
plt.scatter(C_3_S, C_3_RGB)
plt.show()

B_1_709 = prettywebsite.brightness.relativeLuminance_BT709(img1)
B_2_709 = prettywebsite.brightness.relativeLuminance_BT709(img2)
B_3_709 = prettywebsite.brightness.relativeLuminance_BT709(img3)

B_1_601 = prettywebsite.brightness.relativeLuminance_BT709(img1)
B_2_601 = prettywebsite.brightness.relativeLuminance_BT709(img2)
B_3_601 = prettywebsite.brightness.relativeLuminance_BT709(img3)

plt.figure("709 vs 601")
plt.title("Brightness estimation")
plt.xlabel("709")
plt.ylabel("601")
plt.scatter(B_1_709, B_1_601)
plt.scatter(B_2_709, B_2_601)
plt.scatter(B_3_709, B_3_601)
plt.show()

print("In Img1 there is/are",len(prettywebsite.faceDetection.getFaces(img1)), "faces")
print("In Img2 there is/are",len(prettywebsite.faceDetection.getFaces(img2)), "faces")
print("In Img3 there is/are",len(prettywebsite.faceDetection.getFaces(img3)), "faces")

CS_1 = prettywebsite.colorDetection.getColorsW3C(img1,plot=True)
for color in CS_1:
    print(color[0],"is used for the",round(color[1],1),"% of the image")

CS_2 = prettywebsite.colorDetection.getColorsW3C(img2,plot=True)
for color in CS_2:
    print(color[0],"is used for the",round(color[1],1),"% of the image")

CS_3 = prettywebsite.colorDetection.getColorsW3C(img3,plot=True)
for color in CS_3:
    print(color[0],"is used for the",round(color[1],1),"% of the image")

