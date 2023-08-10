import json

with open('C:\\Users\\pabailey\\Documents\\FaceAPITesting\\google.json') as goog_data:
    google = json.load(goog_data)
    #print(google)

with open('C:\\Users\\pabailey\\Documents\\FaceAPITesting\\amazon.json') as aws_data:
    amazon = json.load(aws_data)
    #print(amazon)
    
with open('C:\\Users\\pabailey\\Documents\\FaceAPITesting\\microsoft.json') as azure_data:
    microsoft = json.load(azure_data)
    #print(microsoft)

# GoogleCloud Response
print("// Exterior Polygons")
for i in range(4):
    print(google['faceAnnotations'][i]['fdBoundingPoly'])

print("\n // Interior Polygons")
for i in range(4):
    print(google['faceAnnotations'][i]['boundingPoly'])

# AWS Rekognition
for i in range(3):
    print(amazon['FaceDetails'][i]['BoundingBox'])

# Microsoft Cognitive Services
for i in range(3):
    print(microsoft[i]['faceRectangle'])

print("Amazon Rekognition - Age Detection")
for i in range(3):
    faces = ["John McCarthy", "Ed Fredkin", "Joseph Weizenbaum"]
    print(amazon['FaceDetails'][i]['AgeRange'], "for", faces[i])

print("\n")

print("Microsoft Cognitive Services - Age Detection")
for i in range(3):
    faces = ["Claude Shannon", "John McCarthy", "Ed Fredkin"]
    print(microsoft[i]['faceAttributes']['age'], "for", faces[i])

# Microsoft Face API
print(microsoft[2]['faceAttributes']['emotion'])

# Google Cloud Platform - Vision API
print(google['faceAnnotations'][1]['joyLikelihood'])
print(google['faceAnnotations'][1]['sorrowLikelihood'])
print(google['faceAnnotations'][1]['surpriseLikelihood'])
print(google['faceAnnotations'][1]['angerLikelihood'])

# Amazon Rekognition
print(amazon['FaceDetails'][1]['Emotions'])

print(microsoft[0]['faceLandmarks']['eyebrowLeftOuter'])
print(microsoft[0]['faceLandmarks']['pupilLeft'])
print(microsoft[0]['faceAttributes']['headPose'])
print(microsoft[1]['faceAttributes']['headPose'])
print(microsoft[2]['faceAttributes']['headPose'])

print(google['faceAnnotations'][0]['landmarks'][2])
print(google['faceAnnotations'][0]['landmarks'][0])
print(google['faceAnnotations'][0]['rollAngle'])
print(google['faceAnnotations'][0]['panAngle'])
print(google['faceAnnotations'][0]['tiltAngle'])
print(google['faceAnnotations'][1]['rollAngle'])
print(google['faceAnnotations'][1]['panAngle'])
print(google['faceAnnotations'][1]['tiltAngle'])
print(google['faceAnnotations'][2]['rollAngle'])
print(google['faceAnnotations'][2]['panAngle'])
print(google['faceAnnotations'][2]['tiltAngle'])
print(google['faceAnnotations'][3]['rollAngle'])
print(google['faceAnnotations'][3]['panAngle'])
print(google['faceAnnotations'][3]['tiltAngle'])

print(amazon['FaceDetails'][0]['Landmarks'][7])

