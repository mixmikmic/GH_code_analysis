import os, os.path
import xml.etree.ElementTree as ET

images_path = 'data/images/bboxfiltered/'
bbox_annotations_path = 'data/bbox/Annotation/'

synsets = [f for f in os.listdir(images_path)
                if not os.path.isfile(os.path.join(images_path, f))]

i = 0
for synset in synsets:
    
    dir_path = bbox_annotations_path + synset + "/"
    files = [f for f in os.listdir(dir_path)
                 if os.path.isfile(os.path.join(dir_path, f))]
    
    output_path = 'data/bbox/tsv/' + synset + ".tsv"
    with open(output_path, "w") as f:
        
        print("Processing synset " + str(i) + ": " + synset)
    
        for file in files:
            correction_performed = False

            # Traverse XML tree and get the relevant values
            tree = ET.parse(dir_path + str(file))
            root = tree.getroot()
            
            filename = [child for child in root if child.tag == "filename"][0].text
            size = [child for child in root if child.tag == "size"][0]
            
            width = int([child for child in size if child.tag == "width"][0].text)
            height = int([child for child in size if child.tag == "height"][0].text)
            
            obj = [child for child in root if child.tag == "object"][0]
            # AFAIK there is only one bounding box label per image
            bndbox = [child for child in obj if child.tag == "bndbox"][0]
            xmin = int([child for child in bndbox if child.tag == "xmin"][0].text)
            ymin = int([child for child in bndbox if child.tag == "ymin"][0].text)
            xmax = int([child for child in bndbox if child.tag == "xmax"][0].text)
            ymax = int([child for child in bndbox if child.tag == "ymax"][0].text)
            
            # This is to correct for out-of-bounds bounding boxes
            
            if xmax > width:
                xmax = min(xmax, width)
                correction_performed = True    
            if ymax > height:
                ymax = min(ymax, height)
                correction_performed = True
            if xmin < 0:
                xmin = max(xmin, 0)
                correction_performed = True
            if ymin < 0:
                ymin = max(ymin, 0)
                correction_performed = True
            
            if correction_performed:
                print("  WARN: corrected oob bbox.")
            
            output_str = "".join([filename + ".JPEG", 
                                   "\t", str(width),
                                   "\t", str(height),
                                   "\t", str(xmin), 
                                   "\t", str(ymin), 
                                   "\t", str(xmax), 
                                   "\t", str(ymax), 
                                   "\n"])
            f.write(output_str)
    
    i+=1
    
print("All done.")

# Output as json instead

import os
import json

INPUT_DIR = "data/bbox/tsv/"
OUTPUT_DIR = "data/bbox/json/"

files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

for f in files:
    print("Processing: " + f)
    filepath = os.path.join(INPUT_DIR, f)
    dic = {}
    with open(filepath) as fp:
        data = fp.readlines()
        data = [x.strip() for x in data]
        for item in data:
            ls = item.split('\t')
            key = ls[0]
            value = list(map(int, ls[1:]))
            dic[key] = value
        
    with open(os.path.join(OUTPUT_DIR, f.split('.')[0] + ".json"), 'w') as fp:
        json.dump(dic, fp)
        
print("All done.")

# Confirm the one-bounding box assumption

import os, os.path
import xml.etree.ElementTree as ET

bbox_annotations_path = 'data/bbox/Annotation/'

synsets = [f for f in os.listdir(bbox_annotations_path)
                if not os.path.isfile(os.path.join(bbox_annotations_path, f))]

i = 0
mbbox_count = 0
for synset in synsets:
    
    dir_path = bbox_annotations_path + synset + "/"
    files = [f for f in os.listdir(dir_path)
                 if os.path.isfile(os.path.join(dir_path, f))]
        
    print("Processing synset " + str(i) + ": " + synset)
    
    for file in files:
        correction_performed = False

        # Traverse XML tree and get the relevant values
        tree = ET.parse(dir_path + str(file))
        root = tree.getroot()

        filename = [child for child in root if child.tag == "filename"][0].text
        size = [child for child in root if child.tag == "size"][0]

        width = int([child for child in size if child.tag == "width"][0].text)
        height = int([child for child in size if child.tag == "height"][0].text)

        obj = [child for child in root if child.tag == "object"][0]
        # AFAIK there is only one bounding box label per image
        bndboxes = [child for child in obj if child.tag == "bndbox"]
        if len(bndboxes) > 1:
            print(" WARN: More than one bounding box")
            mbbox_count += 1
    
    i+=1
    
print(str(mbbox_count))



