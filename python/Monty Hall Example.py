import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

# Big enough for accuracy
N = 100000

# The car position is set randomly in any of the 3 doors
car_position = np.random.randint(3, size = N) + 1

# The first selection is choosen randomly for any of the 3 doors
first_selection = np.random.randint(3, size = N) + 1

# Plot hists
f, ax = plt.subplots(ncols=2, nrows=1, figsize = (10,3))
ax[0].hist(car_position)
ax[0].set_title("Distribition of car position")
ax[1].hist(first_selection)
ax[1].set_title("Distribition of first selection")
plt.show()

win_count_random = 0
win_count_stay = 0
win_count_change = 0
for i in range(len(car_position)):
    doors = [1,2,3]
    
    ## Remove selection and car
    doors.remove(first_selection[i])
    if car_position[i] in doors:
        doors.remove(car_position[i])
    
    ## If selection and car are the same chose randomly the door to open
    door_opened = doors[np.random.randint(len(doors))]
    
    ## The new options are my first plus the remaining
    new_options = [1,2,3]
    new_options.remove(door_opened)
    
    ## Select randomly between the 2 options
    second_selection_random = new_options[np.random.randint(2)]
    
    ## Change selection
    new_options.remove(first_selection[i])
    change_door = new_options[0]
    
    if  car_position[i] == second_selection_random:
        win_count_random = win_count_random + 1
        
    if  car_position[i] == first_selection[i]:
        win_count_stay = win_count_stay + 1
        
    if  car_position[i] == change_door:
        win_count_change = win_count_change + 1
        
print("Select second time ramdomly: %d"%(100*win_count_random/N))
print("Select the same door: %d"%(100*win_count_stay/N))
print("Change door: %d"%(100*win_count_change/N))





