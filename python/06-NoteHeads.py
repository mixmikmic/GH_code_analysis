get_ipython().magic('matplotlib inline')
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Image loading, binarization, inversion and display
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin_otsu(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    plt.figure()
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()
    
# Image morphological operations
def dilate(image, kernel, iterations = 1):
    return cv2.dilate(image, kernel, iterations)
def erode(image, kernel, iterations = 1):
    return cv2.erode(image, kernel, iterations)
def open_image(image, kernel = None):
    if kernel is None:
        kernel = np.ones((1, 100))
    return dilate(erode(image, kernel), kernel)

# Horizontal projection
def horizontal_projection(image):
    hor_proj = []
    for i in range(len(image)):
        row_sum = 0
        for j in range(len(image[i])):
            row_sum += image[i][j] == 255
        hor_proj.append([255] * row_sum + [0] * (len(image[0]) - row_sum))

    return hor_proj

# Image crop
def crop_image(image, crop_start = None, crop_width = None):
    if crop_width is None:
        crop_width = len(image[0]) // 10
        
    if crop_start is None:
        end = 0
        for row in image:
            s = sum(row) / 255
            if s > end:
                end = s

        crop_start = end - crop_width
        
    cutoff = image[:]
    
    for i in range(len(cutoff)):
        cutoff[i] = cutoff[i][crop_start : crop_start + crop_width] 

    cutoff = np.array(cutoff, dtype = np.uint8)
    return cutoff

# Find Y coordinates of white pixels
def find_y(image):
    y = []
    for i in range(len(image)):
        for j in range(len(image[i])):
            if (image[i][j] == 255) and (i not in y):
                y.append(i)
    return sorted(y)

# Intersect two lists
def intersect_lists(first, second):
    ret_val = []
    for val in first:
        if val in second:
            ret_val += [val]
    return ret_val

# Group points and get distances
def label_y(y_list):
    labels = [[]]
    line_distances = []
    prev_y = None
    for y in y_list:
        if prev_y is not None:
            if y - prev_y > 1:
                labels.append([])
                line_distances += [y - prev_y]
        labels[-1] += [y]
        prev_y = y
    return labels, line_distances

# Find lines
def find_lines(image):
    first = find_y(crop_image(horizontal_projection(image)))
    second = find_y(open_image(image))
    
    
    lines, distances = label_y(intersect_lists(first, second))
    staff_spacings = [distances[i] for i in range(len(distances)) if (i+1) % 5 != 0 ]
    staff_spacing = sum(staff_spacings) * 1./len(staff_spacings)
    return lines, distances, staff_spacing

# Remove lines
def remove_lines(org_image, tolerance = 0, lines = None, topBotPixelRemoval = True, widthBasedRemoval = True):
    image = org_image.copy()
    
    if lines == None:
        lines, distances, staff_spacing = find_lines(org_image)
    
    if topBotPixelRemoval:
        for line in lines:
            top = line[0]
            bot = line[-1]
            for j in range(len(image[top])):
                remove = True
                is_line = False
                for row in image[top:bot+1]:
                    if row[j] == 255:
                        is_line = True
                        break
                if not is_line:
                    continue
                # check 2 pixels above and below
                diff = 2
                for row in image[top - diff : top]:
                    if row[j] == 255:
                        remove = False
                        break
                if remove:
                    for row in image[bot + 1: bot + diff + 1]:
                        if row[j] == 255:
                            remove = False
                            break
                if remove:
                    for row in image[top:bot+1]:
                        row[j] = 0
    
    if widthBasedRemoval:
        avg_thickness = lines[:]
        for i, line in enumerate(avg_thickness):
            avg_thickness[i] = len(line)
        avg_thickness = sum(avg_thickness) * 1./len(avg_thickness)

        for j in range(len(image[0])):
            white = False
            for i in range(len(image)):
                if image[i][j] == 255:
                    if not white:
                        start = i
                    white = True
                else:
                    if white:
                        thickness = i - start
                        if thickness <= (avg_thickness + tolerance):
                            for row in image[start : i]:
                                row[j] = 0
                    white = False
    return image

def add_region(image, row, col, regions):
    append = True
    coords = [(row, col)]
    idx = 0
    while (idx < len(coords)):
        row, col = coords[idx]
        for dr in range(-1,2):
            for dc in range(-1,2):
                r = row + dr
                c = col + dc
                if r >= 0 and c >= 0 and r < len(image) and c < len(image[r]):
                    if image[r][c] == 255 and ((r,c) not in coords):
                        skip = False
                        for region in regions:
                            if (r,c) in region:
                                skip = True
                                append = False
                                for coord in coords:
                                    region.append((r,c))
                        if not skip:
                            coords += [(r,c)]
        idx += 1
    if append:
        regions.append(coords)

def find_vertical_lines(image):
    # Find lines, distances
    lines, distances, staff_spacing = find_lines(image)

    # Find vertical objects
    img_open = open_image(remove_lines(image), np.ones((1.5 * staff_spacing, 1)))
    return img_open

def find_regions(org_image, ref_image = None):
    if ref_image is None:
        ref_image = org_image
    # Label regions of interest
    regions = []
    for row in range(len(ref_image)):
        for col in range(len(ref_image[row])):
            if ref_image[row][col] == 255:
                isFound = False
                for region in regions:
                    if (row,col) in region:
                        isFound = True
                        break
                if not isFound:
                    add_region(org_image, row, col, regions)
    
    img_regions = org_image.copy()
    for row in range(len(img_regions)):
        for col in range(len(img_regions[row])):
            img_regions[row][col] = 0

    for region in regions:
        for row, col in region:
            img_regions[row, col] = 255
            
    return img_regions, regions

def find_vertical_objects(image, image_vert_lines):
    return find_regions(image, image_vert_lines)

def split_image(image, regions):
    split_images = []
    for region in regions:
        minr = min([r for r,c in region])
        maxr = max([r for r,c in region])
        minc = min([c for r,c in region])
        maxc = max([c for r,c in region])
        sub_image = []
        for row in range(minr,maxr+1):
            sub_image.append([])
            for col in range(minc,maxc+1):
                sub_image[-1] += [image[row][col]]
        sub_image = np.array(sub_image)
        sub_image = np.uint8(sub_image)
        split_images.append(sub_image)
    return split_images

def resize_image(tmp_img, new_width, new_height):
    return cv2.resize(tmp_img, (int(round(new_width)), int(round(new_height))), interpolation = cv2.INTER_CUBIC)

def match_clef(obj, clef_templates):
    obj_height, obj_width = obj.shape[:2]
    best_match = (None, 0)
    for template in clef_templates:
        template_name = template
        # Template Image Processing
        template = load_image(template)
        template = resize_image(template,obj_width,obj_height)
        template = image_gray(template)
        template = image_bin_otsu(template)
        template = invert(template)
        match = 0
        for row in range(len(template)):
            for col in range(len(template[row])):
                match += 1 if obj[row][col] == template[row][col] else 0


        # Normalize
        match *= 1./(obj_width * obj_height)
        if match > best_match[1]:
            best_match = (template_name, match)
    print("Best match: %d%%" % (best_match[1]*100))
    print("Template name: %s" % best_match[0])
    return best_match

org_image = load_image("test_images/staff-with-notes.jpg")
img_gray = image_gray(org_image)
img_otsu = image_bin_otsu(img_gray)
inv_img = invert(img_otsu)
img_wo_lines = remove_lines(inv_img)
display_image(img_wo_lines)

img_vert_lines = find_vertical_lines(inv_img)
display_image(img_vert_lines)

img_vert_objects, regions = find_vertical_objects(img_wo_lines, img_vert_lines)
display_image(img_vert_objects)
print("Number of recognized objects: %s" % len(regions))

from os import listdir
from os.path import isdir
g_clef_templates = []

vertFile = "clefs/g_clef"
split = vertFile.split('/')
for listedFile in listdir("templates"):
    if listedFile == split[0]:
        for innerFile in listdir("templates/%s" % listedFile):
            if len(split) == 1 or innerFile.startswith(split[1]):
                g_clef_templates += ["templates/%s/%s" % (listedFile, innerFile)]

print(g_clef_templates)

objects = split_image(img_vert_objects, regions)
        
best_obj = (None, (None, 0))
best_index = None
for i in range(len(objects)):
    obj = objects[i]
    match = match_clef(obj, g_clef_templates)
    if match[1] > best_obj[1][1]:
        best_obj = (obj, match)
        best_index = i
display_image(best_obj[0])

for r,c in regions[best_index]:
    img_vert_objects[r][c] = 0
display_image(img_vert_objects)
regions.remove(regions[best_index])
objects.remove(objects[best_index])
print(len(regions))

def search_for_templates(vertFile):
    templates = []
    split = vertFile.split('/')
    for listedFile in listdir("templates"):
        if listedFile == split[0]:
            for innerFile in listdir("templates/%s" % listedFile):
                if len(split) == 1 or innerFile.startswith(split[1]):
                    templates += ["templates/%s/%s" % (listedFile, innerFile)]
    return templates

filled_head_templates = search_for_templates("note_heads/filled")
print(filled_head_templates)

lines, distances, staff_spacing = find_lines(inv_img)
print(staff_spacing)
templates = {}
for templateName in filled_head_templates:
    template = load_image(templateName)
    template = resize_image(template,staff_spacing,staff_spacing)
    template = image_gray(template)
    template = image_bin_otsu(template)
    template = invert(template)
    templates[templateName] = template

note = objects[0]
object_height, object_width = note.shape[:2]
best_match = (None,(0,0),0)
for templateName, template in templates.items():
    print("Template matching: %s" % templateName)
    match_matrix = []
    for row in range(object_height - len(template)):
        match_matrix.append([])
        for col in range(object_width - len(template[0])):
            match = 0
            for r in range(len(template)):
                for c in range(len(template[r])):
                    match += 1 if note[row + r ][col + c] == template[r][c] else 0
            match *= 1./(len(template) * len(template[0]))
            
            match_matrix[-1] += [match]
            if match > best_match[2]:
                best_match = (templateName,(row,col),match)
                
# Normalize
print("best match: %d%%" % (best_match[2]*100))
print("templateName: %s" % best_match[0])
print("rows: %s - %s" % (best_match[1][0], best_match[1][0] + len(templates[best_match[0]])))
print("cols: %s - %s" % (best_match[1][1], best_match[1][1] + len(templates[best_match[0]][0])))

template = templates[best_match[0]]
print("Segmented object:")
display_image(note)
recognized_part = note[best_match[1][0]:][best_match[1][1]:best_match[1][1] + len(templates[best_match[0]][0])]
print("Note head of segmented object:")
display_image(recognized_part)
print("height of segment: %s, width of segment: %s" % (len(recognized_part), len(recognized_part[0])))
print("")
print("Best match template:")
display_image(template)
print("height of template: %s, width of template: %s" % (len(template), len(template[0])))

def match_object(note, templates):
    object_height, object_width = note.shape[:2]
    best_match = (None,(0,0),0)
    for templateName, template in templates.items():
        match_matrix = []
        for row in range(object_height - len(template) + 1):
            match_matrix.append([])
            for col in range(object_width - len(template[0]) + 1):
                match = 0
                for r in range(len(template)):
                    for c in range(len(template[r])):
                        match += 1 if note[row + r ][col + c] == template[r][c] else 0
                match *= 1./(len(template) * len(template[0]))

                match_matrix[-1] += [match]
                if match > best_match[2]:
                    best_match = (templateName,(row,col),match)

    if best_match[0] is None:
        print("NO MATCH!")
    else:
        print("best match: %d%%" % (best_match[2]*100))
        print("templateName: %s" % best_match[0])
        print("rows: %s - %s" % (best_match[1][0], best_match[1][0] + len(templates[best_match[0]])))
        print("cols: %s - %s" % (best_match[1][1], best_match[1][1] + len(templates[best_match[0]][0])))

import time
start = time.clock()
for note in objects:
    match_object(note, templates)
end = time.clock()
print("Elapsed time: %ss" % (end - start))

img_stemless = img_vert_objects.copy()
for row in range(len(img_stemless)):
    for col in range(len(img_stemless[row])):
        if img_vert_lines[row,col] == 255:
            img_stemless[row,col] = 0
display_image(img_stemless)

img, regions = find_regions(img_stemless)
display_image(img)
print("Number of recognized regions: %s" % len(regions))
print("False regions:")
for region in regions:
    if len(region) < 10:
        print region

n = int(staff_spacing / 4.)
y,x = np.ogrid[-n : n +1, -n : n+1]
mask = x*x+y*y <= n*n
kernel = np.zeros((len(mask), len(mask)))
kernel[mask] = 1
kernel = np.uint8(kernel)
img_stemless = open_image(img_stemless, kernel)
img, regions = find_regions(img_stemless)
display_image(img)
print("Number of recognized regions: %s" % len(regions))

objects = split_image(img_stemless, regions)
print("First note head")
display_image(objects[0])

start = time.clock()
for note in objects:
    match_object(note, templates)
end = time.clock()
print("Elapsed time: %ss" % (end - start))

maxy = 0
miny = len(img_stemless)
lowest_region = None
highest_region = None
for region in regions:
    for r,c in region:
        if r > maxy:
            maxy = r
            lowest_region = region
        if r < miny:
            miny = r
            highest_region = region

if any([ r == lines[0][0] or r > lines[0][0] for r,c in highest_region]):
    print("Highest region is either around the highest staff line or under it!")
else:
    print("Highest region is above the highest staff line!")
if any([ r == lines[-1][-1] or r < lines[-1][-1] for r,c in lowest_region]):
    print("Lowest region is either around the lowest staff line or above it!")
else:
    print("Highest region is below the lowest staff line!")

two_thirds_spacing = round(staff_spacing * 2. / 3)
from collections import OrderedDict
sub_images = OrderedDict()
for i in range(len(lines)):
    if i == 0:
        sub_images["Above Line %s" % (i + 1)] = img_stemless[lines[i][0] - int(staff_spacing) - len(lines[i]) : lines[i][-1]]
    sub_images["On Line %s" % (i + 1)] = img_stemless[lines[i][0] -  int (two_thirds_spacing) : lines[i][-1] + int(two_thirds_spacing)]
    if i + 1 < len(lines):
        sub_images["Below Line %s" % (i + 1)] = img_stemless[lines[i][0] : lines[i + 1][-1]]
    else:
        sub_images["Below Line %s" % (i + 1)] = img_stemless[lines[i][0] : lines[i][-1] + int(staff_spacing) + len(lines[i])]

for key, image in sub_images.items():
    print(key)
    display_image(image)

start = time.clock()
for key, sub_image in sub_images.items():
    print(key)
    match_object(sub_image, templates)
end = time.clock()
print("Elapsed time: %ss" % (end - start))

start = time.clock()
for key, sub_image in sub_images.items():
    print(key)
    img, regions = find_regions(sub_image)
    objects = split_image(img, regions)
    for obj in objects:
        match_object(obj, templates)
end = time.clock()
print("Elapsed time: %ss" % (end - start))

org_image = load_image("test_images/noteheads.jpg")
img_gray = image_gray(org_image)
img_otsu = image_bin_otsu(img_gray)
inv_img = invert(img_otsu)
img_wo_lines = remove_lines(inv_img, topBotPixelRemoval = True, widthBasedRemoval = False)
display_image(img_wo_lines)
lines, distances, staff_spacing = find_lines(inv_img)

img_vert_lines = find_vertical_lines(inv_img)
display_image(img_vert_lines)

img_vert_objects, regions = find_vertical_objects(img_wo_lines, img_vert_lines)
display_image(img_vert_objects)
print("Number of recognized objects: %s" % len(regions))

img_stemless = img_vert_objects.copy()
for row in range(len(img_stemless)):
    for col in range(len(img_stemless[row])):
        if img_vert_lines[row,col] == 255:
            img_stemless[row,col] = 0
display_image(img_stemless)

two_thirds_spacing = round(staff_spacing * 2. / 3)
from collections import OrderedDict
sub_images = OrderedDict()
for i in range(len(lines)):
    if i == 0:
        sub_images["Above Line %s" % (i + 1)] = img_stemless[lines[i][0] - int(staff_spacing) - len(lines[i]) : lines[i][-1]]
    sub_images["On Line %s" % (i + 1)] = img_stemless[lines[i][0] -  int (two_thirds_spacing) : lines[i][-1] + int(two_thirds_spacing)]
    if i + 1 < len(lines):
        sub_images["Below Line %s" % (i + 1)] = img_stemless[lines[i][0] : lines[i + 1][-1]]
    else:
        sub_images["Below Line %s" % (i + 1)] = img_stemless[lines[i][0] : lines[i][-1] + int(staff_spacing) + len(lines[i])]
for key, image in sub_images.items():
    print(key)
    display_image(image)

note_heads_templates = search_for_templates("note_heads")
templates = {}
for templateName in note_heads_templates:
    template = load_image(templateName)
    template = resize_image(template,staff_spacing,staff_spacing)
    template = image_gray(template)
    template = image_bin_otsu(template)
    template = invert(template)
    templates[templateName] = template

print("On Line 4")
img, regions = find_regions(sub_images["On Line 4"])
objects = split_image(img, regions)
for obj in objects:
    display_image(obj)
    match_object(obj, templates)

img_vert_objects, regions = find_vertical_objects(img_wo_lines, img_vert_lines)
display_image(img_vert_objects)
print("Number of recognized objects: %s" % len(regions))

img, regions = find_regions(img_vert_objects)
objects = split_image(img, regions)
display_image(objects[0])

note_templates = search_for_templates("notes/2")
templates = {}
for templateName in note_templates:
    template = load_image(templateName)
    template = resize_image(template,staff_spacing,staff_spacing)
    template = image_gray(template)
    template = image_bin_otsu(template)
    template = invert(template)
    templates[templateName] = template
    template = np.rot90(template, 2)
    templates[templateName + "_rotated"] = template

match_object(objects[0], templates)

img, regions = find_regions(img_wo_lines)
objects = split_image(img, regions)
display_image(objects[7])

# we take only whole note heads, because we already got filled and half note heads
note_heads_templates = search_for_templates("note_heads/whole")
templates = {}
for templateName in note_heads_templates:
    template = load_image(templateName)
    template = resize_image(template,staff_spacing,staff_spacing)
    template = image_gray(template)
    template = image_bin_otsu(template)
    template = invert(template)
    templates[templateName] = template

match_object(objects[7], templates)



