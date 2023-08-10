import os
import numpy as np
import pandas as pd
import shutil

# We manually scanned over appx 20000 property and made sure to the best of my knowledge that they allign with the label

# We reviewed it or Aerial Cropped images and would sync the same for Aerial
path_to_aerial_cropped = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial_cropped'
old_aerial_dump_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial'
new_aerial_dump_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\new_aerial'

cropped_land_path = os.path.join(path_to_aerial_cropped, 'land')
cropped_house_path = os.path.join(path_to_aerial_cropped, 'house')

aerial_cropped_land_pins = [p.split('.')[0] for p in os.listdir(cropped_land_path) if p not in '.DS_Store']
aerial_cropped_house_pins = [p.split('.')[0] for p in os.listdir(cropped_house_path) if p not in '.DS_Store']

old_aerial_land_pins = [p.split('.')[0] for p in os.listdir(os.path.join(old_aerial_dump_path, 'land')) if p not in '.DS_Store']
old_aerial_house_pins = [p.split('.')[0] for p in os.listdir(os.path.join(old_aerial_dump_path, 'house')) if p not in '.DS_Store']
    
new_aerial_land_pins = [p.split('.')[0] for p in os.listdir(os.path.join(new_aerial_dunp_path, 'land')) if p not in '.DS_Store']
new_aerial_house_pins = [p.split('.')[0] for p in os.listdir(os.path.join(new_aerial_dunp_path, 'house')) if p not in '.DS_Store']

print (len(aerial_cropped_land_pins))
print (len(aerial_cropped_house_pins))
print (len(old_aerial_land_pins))
print (len(old_aerial_house_pins))

good_lands = np.intersect1d(aerial_cropped_land_pins, old_aerial_land_pins)
print (len(good_lands))

not_found = 0
for pins in good_lands:
    from_path = os.path.join(old_aerial_dump_path, 'land', pins+'.jpg')
    to_path = os.path.join(new_aerial_dump_path, 'land', pins+'.jpg')
#     print(from_path)
#     print(to_path)
#     break
    if os.path.exists(from_path):
        shutil.move(from_path, to_path)
    else:
        not_found+=1
print (not_found)

good_house = np.intersect1d(aerial_cropped_house_pins, old_aerial_house_pins)
print (len(good_house))

not_found = 0
for pins in good_house:
    from_path = os.path.join(old_aerial_dump_path, 'house', pins+'.jpg')
    to_path = os.path.join(new_aerial_dump_path, 'house', pins+'.jpg')
#     print(from_path)
#     print(to_path)
#     break
    if os.path.exists(from_path):
        shutil.move(from_path, to_path)
    else:
        not_found+=1
print (not_found)

mislabeled_land = np.intersect1d(aerial_cropped_land_pins, old_aerial_house_pins)
print (len(mislabeled_land))

not_found = 0
for pins in mislabeled_land:
    from_path = os.path.join(old_aerial_dump_path, 'house', pins+'.jpg')
    to_path = os.path.join(new_aerial_dump_path, 'land', pins+'.jpg')
#     print(from_path)
#     print(to_path)
#     break
    if os.path.exists(from_path):
        shutil.move(from_path, to_path)
    else:
        not_found+=1
print (not_found)

mislabeled_house = np.intersect1d(aerial_cropped_house_pins, old_aerial_land_pins)
print (len(mislabeled_house))

not_found = 0
for pins in mislabeled_house:
    from_path = os.path.join(old_aerial_dump_path, 'land', pins+'.jpg')
    to_path = os.path.join(new_aerial_dump_path, 'house', pins+'.jpg')
#     print(from_path)
#     print(to_path)
#     break
    if os.path.exists(from_path):
        shutil.move(from_path, to_path)
    else:
        not_found+=1
print (not_found)

aerial_house_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial\house'
aerial_land_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial\land'
house_pins = [p.split('.')[0] for p in os.listdir(aerial_house_path) if p not in '.DS_Store']
land_pins = [p.split('.')[0] for p in os.listdir(aerial_land_path) if p not in '.DS_Store']

house_land = house_pins + land_pins
house_land_indicator = np.append(np.tile('Likely House', len(house_pins)), np.tile('Likely Land', len(land_pins)))
print(len(house_land), len(house_land_indicator))
pd.DataFrame({'pin':house_land, 'indicator':house_land_indicator}).to_csv(r'C:\Users\newline\Documents\ImageClassification\data\actual_land_house_list.csv')

old_pin_metadata_list = r'C:\Users\newline\Documents\ImageClassification\data\statistics\sam_new\aerial\aerial_collected_data_stats\1524181313_0_29234.csv'
new_pin_metadata_list = r'C:\Users\newline\Documents\ImageClassification\data\actual_land_house_list.csv'
old_metadata = pd.read_csv(old_pin_metadata_list)
new_metadata = pd.read_csv(new_pin_metadata_list)
print (old_metadata.shape, new_metadata.shape)
old_metadata.head()

print (len(np.unique(new_metadata[['pin']])), len(new_metadata))
new_metadata.head()

# we change the old indicator to original_indicator place a new indicator
new_metadata = new_metadata.drop('Unnamed: 0', axis=1)
new_metadata.dtypes

old_metadata = old_metadata.drop('indicator', axis=1)
old_metadata.dtypes

new_corrected_meta = new_metadata.merge(old_metadata, how='inner',  on='pin')
print(new_corrected_meta.shape)
new_corrected_meta.head()

# Dump the corrected Data
new_corrected_meta.to_csv(old_pin_metadata_list, index=None)

aerial_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial'
aerial_cropped_path = r'C:\Users\newline\Documents\ImageClassification\data\input_images\sam_new\aerial_cropped'

aerial_land_pins = [p.split('.')[0] for p in os.listdir(os.path.join(aerial_path, 'land')) if p != '.DS_Store']
aerial_house_pins = [p.split('.')[0] for p in os.listdir(os.path.join(aerial_path, 'house')) if p != '.DS_Store']

aerial_cropped_land_pins = [p.split('.')[0] for p in os.listdir(os.path.join(aerial_cropped_path, 'land')) if p != '.DS_Store']
aerial_cropped_house_pins = [p.split('.')[0] for p in os.listdir(os.path.join(aerial_cropped_path, 'house')) if p != '.DS_Store']

print (len(aerial_land_pins), len(aerial_cropped_land_pins), len(np.intersect1d(aerial_land_pins, aerial_cropped_land_pins)))
print (len(aerial_house_pins), len(aerial_cropped_house_pins), len(np.intersect1d(aerial_house_pins, aerial_cropped_house_pins)))



