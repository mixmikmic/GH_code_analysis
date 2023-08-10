import py_wsi

file_dir = "/Users/ysbecca/ysbecca-projects/py_wsi/py_wsi/wsi_data/"
xml_dir = file_dir
patch_size = 256
level = 12
db_location = ""
db_name = "patch_db"
overlap = 0

# All possible labels mapped to integer ids in order of increasing severity.
label_map = {'Normal': 0,
#              'Benign': 1,
             'Carcinoma in situ': 2,
             'In situ carcinoma': 2,
             'Carcinoma invasive': 3,
             'Invasive carcinoma': 3,
            }

turtle = py_wsi.Turtle(file_dir, db_location, db_name, xml_dir=xml_dir, label_map=label_map)

print("Total WSI images:    " + str(turtle.num_files))
print("LMDB name:           " + str(turtle.db_name))
print("File names:          " + str(turtle.files))
print("XML files found:     " + str(turtle.get_xml_files()))

turtle.sample_and_store_patches(patch_size, level, overlap, load_xml=True)

import dataset as ds
import imagepy_toolkit

dataset = ds.read_datasets(turtle,
                set_id=1,
                valid_id=0,
                total_sets=5,
                shuffle_all=True,
                augment=False)

print("Total training set images:     " + str(len(dataset.train.images)))
print("Total validation set images:   " + str(len(dataset.valid.images)))

imagepy_toolkit.show_images(dataset.train.images, 5, 1)

imagepy_toolkit.show_labeled_patches(dataset.train.images, dataset.train.image_cls)





