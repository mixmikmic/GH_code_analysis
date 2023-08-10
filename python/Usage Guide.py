import Automold as am
import Helpers as hp

path='./test_augmentation/*.jpg'
images= hp.load_images(path)

hp.visualize(images, column=3, fig_size=(20,10))

bright_images= am.brighten(images[0:3]) ## if brightness_coeff is undefined brightness is random in each image
hp.visualize(bright_images, column=3)

bright_images= am.brighten(images[0:3], brightness_coeff=0.7) ## brightness_coeff is between 0.0 and 1.0
hp.visualize(bright_images, column=3)

dark_images= am.darken(images[0:3]) ## if darkness_coeff is undefined darkness is random in each image
hp.visualize(dark_images, column=3)

dark_images= am.darken(images[0:3], darkness_coeff=0.7) ## darkness_coeff is between 0.0 and 1.0
hp.visualize(dark_images, column=3)

dark_bright_images= am.random_brightness(images[4:7]) 
hp.visualize(dark_bright_images, column=3)

shadowy_images= am.add_shadow(images[4:7]) 
hp.visualize(shadowy_images, column=3)

shadowy_images= am.add_shadow(images[4:7], no_of_shadows=2, shadow_dimension=8) 
hp.visualize(shadowy_images, column=3)

snowy_images= am.add_snow(images[4:7]) ##randomly add snow
hp.visualize(snowy_images, column=3)

snowy_images= am.add_snow(images[4:7], snow_coeff=0.3) 
hp.visualize(snowy_images, column=3)

snowy_images= am.add_snow(images[4:7], snow_coeff=0.8) 
hp.visualize(snowy_images, column=3)

rainy_images= am.add_rain(images[4:7]) 
hp.visualize(rainy_images, column=3)

rainy_images= am.add_rain(images[4:7], rain_type='heavy', slant=20) 
hp.visualize(rainy_images, column=3)

rainy_images= am.add_rain(images[4:7], rain_type='torrential') 
hp.visualize(rainy_images, column=3)

foggy_images= am.add_fog(images[4:7]) 
hp.visualize(foggy_images, column=3)

foggy_images= am.add_fog(images[4:7], fog_coeff=0.4) 
hp.visualize(foggy_images, column=3)

foggy_images= am.add_fog(images[4:7], fog_coeff=0.9) 
hp.visualize(foggy_images, column=3)

bad_road_images= am.add_gravel(images[4:7]) 
hp.visualize(bad_road_images, column=3)

bad_road_images= am.add_gravel(images[4:7], rectangular_roi=(700,550,1280,720),no_of_patches=20) ##too much gravels on right
hp.visualize(bad_road_images, column=3)

flare_images= am.add_sun_flare(images[4:7]) 
hp.visualize(flare_images, column=3)

import math
flare_images= am.add_sun_flare(images[4:7], flare_center=(100,100), angle=-math.pi/4) ## fixed src center
hp.visualize(flare_images, column=3)

speedy_images= am.add_speed(images[1:4]) ##random speed
hp.visualize(speedy_images, column=3)

speedy_images= am.add_speed(images[1:4], speed_coeff=0.9) ##random speed
hp.visualize(speedy_images, column=3)

fall_images= am.add_autumn(images[0:3]) 
hp.visualize(fall_images, column=3)

flipped_images= am.fliph(images[0:3]) 
hp.visualize(flipped_images, column=3)

flipped_images= am.flipv(images[0:3]) 
hp.visualize(flipped_images, column=3)

flipped_images= am.random_flip(images[0:3]) 
hp.visualize(flipped_images, column=3)

manhole_images= am.add_manhole(images[0:3]) 
hp.visualize(manhole_images, column=3)

aug_images= am.augment_random(images[4:6], volume='same')  ##2 random augmentations applied on both images
hp.visualize(aug_images,column=3,fig_size=(20,20))

aug_images= am.augment_random(images[4:6], volume='expand')  ##all aug_types are applied in both images
hp.visualize(aug_images,column=3,fig_size=(20,20))

aug_images= am.augment_random(images[4:6], aug_types=['add_sun_flare','add_speed','add_autumn'], volume='expand')  ##all aug_types are applied in both images
hp.visualize(aug_images,column=3,fig_size=(20,10))

aug_images= am.augment_random(images[4:6], aug_types=['add_sun_flare','add_speed','add_autumn'], volume='same')  ##2 random aug_types are applied in both images
hp.visualize(aug_images,column=3,fig_size=(20,10))

aug_types=["random_brightness","add_shadow","add_snow","add_rain","add_fog","add_gravel","add_sun_flare","add_speed","add_autumn","random_flip","add_manhole"]
dict_time={}
import time
for aug_type in aug_types:
    t=time.time()
    command='am.'+aug_type+'(images)'
    result=eval(command)
    dict_time[aug_type]=time.time()-t
    t=time.time()
print('Average Time taken per augmentaion function to process 1 image:')
tot=0
for key, value in dict_time.items():
    tot+=value
    print(key, '{0:.2f}s'.format(value/len(images)))

print('-----------------------')
print('Total no. of augmented images created:', len(aug_types)*len(images))
print('-----------------------')
print('Total time taken to create ',len(aug_types)*len(images),' augmented images:', '{0:.2f}s'.format(tot))

