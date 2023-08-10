from generator import *

model = load_model()

image, original, filename, filepath = preprocess("test_images/2005-hwy-side_b-5.jpg")

kernel = (11, 11)
iterations = 2

contours = detect_roi(image, kernel, iterations)

sorted_contours = sort_contours(contours)

classified_contours, contour_types = classify(sorted_contours, image, model)

Image(filename="output/image_contours.png")

false_positives = false_positives(raw_input())

updated_contours, updated_contour_types = redraw(image, classified_contours, contour_types, false_positives)

Image(filename="output/image_contours_updated.png")

mark = raw_input()

if mark == 'y':
    updated_contours, updated_contour_types = draw_roi(image, updated_contours, updated_contour_types)
else:
    pass

hires_contours = project(image, original, updated_contours)

generate_annotation(filename, original, hires_contours, updated_contour_types)

