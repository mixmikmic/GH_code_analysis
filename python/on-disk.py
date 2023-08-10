import trackpy as tp
import pims

def gray(image):
    return image[:, :, 0]

images = pims.ImageSequence('../sample_data/bulk_water/*.png', process_func=gray)
images = images[:10]  # We'll take just the first 10 frames for demo purposes.

# For this demo, we'll first remove the file if it already exists.
get_ipython().system('rm -f data.h5')

with tp.PandasHDFStore('data.h5') as s:  # This opens an HDF5 file. Data will be stored and retrieved by frame number.
    for image in images:
        features = tp.locate(image, 11, invert=True)  # Find the features in a given frame.
        s.put(features)  # Save the features to the file before continuing to the next frame.

with tp.PandasHDFStore('data.h5') as s:
    tp.batch(images, 11, invert=True, output=s)

with tp.PandasHDFStore('data.h5') as s:
    frame_2_results = s.get(2)
    
frame_2_results.head()  # Display the first few rows.

with tp.PandasHDFStore('data.h5') as s:
    all_results = s.dump()
    
all_results.head()  # Display the first few rows.

with tp.PandasHDFStore('data.h5') as s:
    for linked in tp.link_df_iter(s, 3, neighbor_strategy='KDTree'):
        s.put(linked)

with tp.PandasHDFStore('data.h5') as s:
    frame_2_results = s.get(2)
    
frame_2_results.head()  # Display the first few rows.

