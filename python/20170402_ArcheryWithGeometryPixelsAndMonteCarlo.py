from PIL import Image
import numpy as np

im = Image.open("TargetMonteCarlo.bmp")
# Convert the image into an array of [R,G,B] per pixel
data = np.array(im.getdata(), np.uint8).reshape(im.size[1], im.size[0], 3)
# For example, the upper left pixel is black ([0,0,0]):
print("Upper left pixel: {}".format(data[0][0]))
# The upper center pixel is blue ([0, 162, 232]):
print("Upper middel pixel: {}".format(data[0][99]))
# And the ~middle pixel is red ([237, 28, 36]):
print("Middle pixel: {}".format(data[99][99]))

# Here we are using a trick: "162" is unique to the blue pixels,
# and "28" is unique to the red pixels, so we can search for it
# alone. Any of the pixels that are left from the total must be
# black, so we can count (length of the data - blue - red) pixels.
blue = len(data[np.where(data == 162)])
red = len(data[np.where(data == 28)])
black = len(im.getdata()) - blue - red
print("Black:{}, Blue:{}, Red:{}".format(black, blue, red))

total = blue + red + black
# The probabilities are the count over the total pixels
p_t = float(blue) / float(total)
p_b = float(red) / float(total)
# To get the expected value, we use the same formula as before:
# e_d = (p_n)(0) + (p_t)(1) + (p_b)(2)
e_d = p_t + 2*p_b
print("Expected score per dart throw:{}".format(round(e_d,3)))

from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Problem constraints
r_b = 0.1
r_t = 1
mean_x = 0.05 * r_t

# Iterate over many possible sigma_y
results = {}
for sigma_x in np.linspace(0.1, 1, 100):
    prob_hit = norm.cdf(r_b, loc=mean_x, scale=sigma_x) -                norm.cdf(-r_b, loc=mean_x, scale=sigma_x)
    results[sigma_x] = prob_hit

# Save the results to a DataFrame and plot them
df = pd.DataFrame.from_dict(results, orient="index")
f = df.plot();
f.set_ylabel("prob_hit")
f.set_xlabel("sigma_x");
plt.scatter(0.385, 0.2, c = 'r');

# Calculate sigma_y based on sigma_x
print('Sigma_y = {}'.format(round(0.385/0.75,3)))

import random
import numpy as np

# User-defined parameters
n_hist = 1000000
# Radius of the target and bullseye
r_t = 1
r_b = 0.1
# Parameters of the random distributions
mean_x = 0.05
mean_y = 0.05
sigma_x = 0.385
sigma_y = 0.513
# Score for hitting the bullseye, target, and nothing
b_score = 2
t_score = 1
n_score = 0

# Monte Carlo loop
total_score = 0
for n in range(0,n_hist):
    x = random.normalvariate(mean_x, sigma_x)
    y = random.normalvariate(mean_y, sigma_y)
    check = np.sqrt(x**2 + y**2)
    # If it hits the bullseye
    if check < r_b:
        total_score += b_score
    # If it hits the target but not the bullseye
    elif check < r_t:
        total_score += t_score
    # If it hits neither the target nor bullseye
    else:
        total_score += n_score
        
# Find the mean of the Monte Carlo simulation
hist_mean = total_score / n_hist
print('Expected score per dart throw: {}'.format(hist_mean))

