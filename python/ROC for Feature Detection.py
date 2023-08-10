import numpy as np

import matplotlib.pyplot as plt
import matplotlib

get_ipython().run_line_magic('matplotlib', 'inline')


from detector import model

threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

for threshold in threshold_list:
    model(image_path='images/target_tennis.jpg', template_path='images/template_tennis.jpg',
          threshold=threshold, analyze=True)

image = plt.imread('images/target_tennis.jpg')
actual_negatives = (image.shape[0] * image.shape[1]) - 9
print('Number of actual negatives:', actual_negatives)

threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

tp_list = [9, 9, 9, 8, 3, 1]
fp_list = [140, 108, 52, 3, 0, 0]
tn_list = [50535 - 149, 50535 - 117, 50535 - 61, 
           50535 - 11, 50535 - 3, 50535 - 1]
fn_list = [0, 0,  0, 1, 6, 10]

def analyze_results(tp_list, fp_list, tn_list, fn_list, threshold_list,
                   actual_negatives, actual_positives):
    
    tpr_list = []
    fpr_list = []
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 8))
    plt.xticks(rotation=60)
    
    for tp, fp, tn, fn, threshold in zip(tp_list, fp_list, 
                                         tn_list, fn_list, 
                                         threshold_list):
        
        if fp == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        tpr = tp / (tp + fn)

        tpr_list.append(tpr)
        fpr_list.append(fpr)
     
        plt.plot(fpr, tpr + 0.05, marker='$%.2f$' % threshold, 
                 ms = 30, color = 'k', 
                label = 'threshold') 
        
    plt.plot(fpr_list, tpr_list, 'o-', color = 'red'); 
    plt.xlabel('False Postive Rate'); plt.ylabel('True Positive Rate');
    plt.title('ROC Curve'); 
    plt.show()

analyze_results(tp_list, fp_list, tn_list, fn_list, threshold_list, 
                actual_negatives=50535, actual_positives=9)

threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

for threshold in threshold_list:
    model(image_path='images/target_tennis.jpg', 
          template_path='images/template_tennis_two.jpg',
          threshold=threshold, analyze=True)

tp_list = [9, 9, 9, 6, 2, 1]
fp_list = [124, 93, 36, 1, 0, 0]
tn_list = [50535 - 133, 50535 - 102, 50535 - 45, 
           50535 - 7, 50535 - 2, 50535 - 1]
fn_list = [0, 0,  0, 3, 9, 10]

analyze_results(tp_list, fp_list, tn_list, fn_list,
                threshold_list, actual_negatives=50535, actual_positives=9)

from PIL import Image
import matplotlib as mpl

def apply_blur_filter(blur_filter, image_path):
    
    # Load in the image
    image = Image.open(image_path)
    
    # Crop to correct size
    image = image.crop(box=(0, 0, 
                       int(image.size[0] / blur_filter.shape[0]) * blur_filter.shape[0], 
                       int(image.size[1] / blur_filter.shape[1]) * blur_filter.shape[1]))
    
    im_array = np.array(image)
    
    # Horizontal and vertical moves, using a stride of filter shape
    h_moves = int(im_array.shape[1] / blur_filter.shape[1])
    v_moves = int(im_array.shape[0] / blur_filter.shape[0])
    
    new_image = np.zeros(shape = im_array.shape)
    
    k = np.sum(blur_filter)
    
    # Iterate through 3 color channels
    for i in range(im_array.shape[2]):
        # Extract the layer and create a new layer to fill in 
        layer = im_array[:, :, i]
        new_layer = np.zeros(shape = layer.shape, dtype='uint8')

        # Left and right bounds are determined by columns
        l_border = 0
        r_border = blur_filter.shape[1]


        # Iterate through the number of horizontal and vertical moves
        for h in range(h_moves):
            # Top and bottom bounds are determined by rows
            b_border = 0
            t_border = blur_filter.shape[0]
            for v in range(v_moves):
                patch = layer[b_border:t_border, l_border:r_border]

                # Take the element-wise product of the patch and the filter
                product = np.multiply(patch, blur_filter)

                # Find the weighted average of the patch
                product = np.sum(product) / k
                new_layer[b_border:t_border, l_border:r_border] = product

                b_border = t_border
                t_border = t_border + blur_filter.shape[0]

            l_border = r_border
            r_border = r_border + blur_filter.shape[1]


        new_image[:, :, i] = 255 * ( (new_layer - np.min(new_layer)) / 
                                    (np.max(new_layer) - np.min(new_layer)) )


    # Convert to correct type for plotting
    new_image = new_image.astype('uint8')
    
    return new_image

gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                            [2, 8, 12, 8, 2],
                            [6, 24, 36, 24, 6],
                            [2, 8, 12, 8, 2],
                            [1, 4, 6, 4, 1]])

blur_tennis = apply_blur_filter(gaussian_kernel, 'images/target_tennis.jpg')
mpl.image.imsave('images/target_tennis_blurred.jpg', blur_tennis)

threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

for threshold in threshold_list:
    model(image_path='images/target_tennis_blurred.jpg', 
          template_path='images/template_tennis.jpg',
          threshold=threshold, analyze=True)

threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

tp_list = [9, 9, 9, 5, 1, 0]
fp_list = [50, 5, 0, 0, 0, 0]
tn_list = [50535 - 59, 50535 - 14, 50535 - 9, 
           50535 - 5, 50535 - 1, 50535]
fn_list = [0, 0, 0, 3, 8, 9]

analyze_results(tp_list, fp_list, tn_list, fn_list,
                threshold_list, actual_negatives=50535, actual_positives=9)

