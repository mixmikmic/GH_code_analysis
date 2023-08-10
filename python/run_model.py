from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

# Object detection imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Matplot lib for plotting
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Define paths to image sub-folders
root_dir = Path.cwd()
images_path = root_dir / '..' / 'test_images'
output_path = root_dir / '..' / 'output_images'

# Frozen graph file
FROZEN_GRAPH = root_dir / 'models' / 'output_inference_graph.pb' / 'frozen_inference_graph.pb'

# Class labels file
PATH_TO_LABELS = root_dir / 'data' / 'label_map.pbtxt'

# Load frozen model graph from file
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(str(FROZEN_GRAPH), 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map from file
label_map = label_map_util.load_labelmap(str(PATH_TO_LABELS))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=4, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print('Image loaded of size (%s, %s)' % (im_width, im_height))
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in list(images_path.glob('*.png')):
            image = Image.open(str(image_path))
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            print('Detected %s objects in image' % len(boxes))
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
            # Save image to output directory
            Image.fromarray(image_np).save(str(output_path / image_path.name))

