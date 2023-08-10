import cv2
from pathlib import Path
from lib_1.PluginLoader import PluginLoader
from lib_1.faces_detect import detect_faces

input_directory="/home/olivier/Desktop/face-swap/daniel_craig/*"
output_directory="/home/olivier/Desktop/face-swap/daniel_craig_test"

def load_filter():
    filter_file = "filter.jpg" # TODO Pass as argument
    if Path(filter_file).exists():
        print('Loading reference image for filtering')
        return FaceFilter(filter_file)

def get_faces(image):
    faces_count = 0
    filterDeepFake = load_filter()
    
    for face in detect_faces(image):
        
        if filterDeepFake is not None and not filterDeepFake.check(face):
            print('Skipping not recognized face!')
            continue
        

        yield faces_count, face

import glob
included_extentions = ['jpg', 'bmp', 'png', 'gif']
image_list = [fn for fn in glob.glob(input_directory) if any(fn.endswith(ext) for ext in included_extentions)]

extractor_name = "Align" # TODO Pass as argument
extractor = PluginLoader.get_extractor(extractor_name)()

try:
    for filename in image_list:
        
        image = cv2.imread(filename)
        for idx, face in get_faces(image):
            resized_image = extractor.extract(image, face, 256)
            output_file = output_directory+"/"+str(Path(filename).stem)
            cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)

except Exception as e:
    print('Failed to extract from image: {}. Reason: {}'.format(filename, e))



