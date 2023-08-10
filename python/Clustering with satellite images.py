import os

#data manipulation
import numpy as np
import pandas as pd

#reading and displying images
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import seaborn as sns

#displying data
from IPython.display import display

#the K-means implementation
from sklearn.cluster import KMeans

#guassaian smoothing
from scipy.ndimage import gaussian_filter

#inline plots
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (20,20)

files = ["Sentinel-2 image on 2018-012-natural.jpg",
         "Sentinel-2 image on 2018-01-12-agric.jpg",
         "Sentinel-2 image on 2018-01-12-urban.jpg",
         "Sentinel-2 image on 2018-01-12-vegetation.jpg"
        ]

names = ["Natural",
         "Agricultural",
         "Urban",
         "Vegetation"]

file_dir = "../data/ghana_data/"

images = [plt.imread(file_dir + file) for file in files]
images = dict(zip(names, images))

smooth_imgs = []

for name in names:
    smooth_imgs.append(gaussian_filter(images[name], sigma = [5,5,0]))
    
smooth_images = dict(zip(names, smooth_imgs))

for name in names:
 
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(images[name])
    axs[0].set_title(name + ": Unprocessed")
    axs[1].imshow(smooth_images[name])
    axs[1].set_title(name + ": Smoothed")
    plt.show()

def cluster_image(groups, img, method = "random"):
    """cluster_image
    Takes an image, represented as a numpy array and attempts to cluster the pixels
    in the image into a specified number of groups.
    By default uses random starting clusters with a specified random seed
    
    Args:
        groups (int): The number of groups to cluster the data into (k)
        img (Array): The image of dimensions (x,y,z) where z is the features to cluster over
        method (String): Initial starting method to use, random by default. 
            See: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    
    Returns:
        cluster_labels (Array): Contains cluster labels for img in a 2-D array of the same size as the first two
            dimensions of img
    
    """
    
    #put into the right shape
    dims = np.shape(img)
    img_matrix = np.reshape(img, (dims[0] * dims[1], dims[2]))
    
    #cluster
    cl = KMeans(n_clusters = groups, init = method)
    img_groups = cl.fit_predict(img_matrix)
    
    #create image
    cluster_groups = np.reshape(img_groups, (dims[0], dims[1]))
    
    return cluster_groups

def cluster_ks(image, ks):
    
    """cluster_ks
    Wrapper for cluster image. Repeats clustering for a range of values.
    
    Args:
        image (Array): The image of dimensions (x,y,z) where z is the features to cluster over
        ks (iterable): integer values of k to cluster with
        
    Returns: 
        (dict): key:value pair where key is k clusters and value is the results in a numpy array 
        
    """
    
    cluster_labels = []
    
    for k in ks:
        
        #get cluster groups
        group_labels = cluster_image(groups = k, img = image)
        cluster_labels.append(group_labels)
    
    
    clusters = [str(k) for k in ks]
    return dict(zip(clusters,cluster_labels))

def plt_results(ks, 
                imgs, 
                smoothed_imgs, 
                img_name,
                file_type = ".png",
                save_dir = "../results/clustering/"
               ):
    
    """plt_results
    
    Plot results from smoothed and unsmoothed images side by side
    
    Args:
        ks (iterable): the value for k used to cluster
        img (dict): cluster results from unsmoothed image
        smoothed_img (dict): cluster results from smoothed image
        img_name (string): name of the image the results are for
        file_type (string): image file extention for saving, must be something that matplotlib can render
        save_dir (string): directory to save the images to
        
    Returns:
        figs (List): the figures created from the results
    """

    figs =[]
    for k in range(3,10):
        fig, axs = plt.subplots(1,2)
        
        im = axs[0].imshow(imgs[str(k)])
        handle = make_legend_handles(img = im,
                                     n_clusters = k
                                    )
        axs[0].legend(handles = handle)
        axs[0].set_title(img_name + ": {} clusters".format(k))
        
        im = axs[1].imshow(smoothed_imgs[str(k)])
        handle = make_legend_handles(img = im,
                             n_clusters = k
                            )
        axs[1].legend(handles = handle)
        axs[1].set_title(img_name + ", smoothed: {} clusters".format(k))


        plt.show()
        
        #get the counts
        img_res_df = calc_counts(imgs[str(k)])
        smooth_img_res_df = calc_counts(smoothed_imgs[str(k)])
        #put them together
        res_df = pd.concat([img_res_df, smooth_img_res_df],
                           axis = 1,
                           keys = ["unsmoothed", "smoothed"]
                          )  
        display(res_df)
        
        if save_dir is not None:
            img_file = save_dir + img_name + "_{}-clusters".format(k) + file_type
            fig.savefig(img_file)
            res_file = save_dir + img_name + "_{}-clusters.csv".format(k) 
            res_df.to_csv(res_file)
        figs.append(fig)
        
    return

def make_legend_handles(img, n_clusters):
    """make_legend_handles
    
    creates handles for use with legend
    
    Args:
        img (ListedColourmap): the image for the legend
        n_clusters (int): number of clusters
    
    """
    #create colours
    colours = [img.cmap(img.norm(cluster)) for cluster in range(n_clusters)]
    #use a list of Patch objects for the handle
    handles = []
    for cluster in range(n_clusters):
        handles.append(pat.Patch(color = colours[cluster], label = "Cluster {}".format(cluster)))
        
    return handles

def calc_counts(results):
    """calc_counts
    
    Computes and returns counts and ratios for number of pixels in the image within each cluster
    
    Args:
        results (Array): 2D array of cluster labels
    Returns
        df (DataFrame): contains the counts, ratios and percentages of each cluster
    
    """
    uni, counts = np.unique(results, return_counts = True)
    df = pd.DataFrame({"cluster": uni,
                  "pixel_count" : counts})

    df["ratio"] = df["pixel_count"].divide(df["pixel_count"].sum())
    df["% total"] = df["ratio"].multiply(100)
    df.set_index("cluster", inplace = True)
    return df

nat_results = cluster_ks(images["Natural"], range(3,10))
nat_smooth_results = cluster_ks(smooth_images["Natural"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = nat_results,
            smoothed_imgs = nat_smooth_results, 
            img_name = "Natural"
           )

agric_results = cluster_ks(images["Agricultural"], range(3,10))
smooth_agric_results = cluster_ks(smooth_images["Agricultural"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = agric_results,
            smoothed_imgs = smooth_agric_results, 
            img_name = "Agricultural"
           )

urban_results = cluster_ks(images["Urban"], range(3,10))
smooth_urb_results = cluster_ks(smooth_images["Urban"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = urban_results,
            smoothed_imgs = smooth_urb_results, 
            img_name = "Urban"
           )

veg_results = cluster_ks(images["Vegetation"], range(3,10))
smooth_veg_results = cluster_ks(smooth_images["Vegetation"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = veg_results,
            smoothed_imgs = smooth_veg_results, 
            img_name = "Vegetation"
           )

all_img = np.concatenate([images["Natural"], 
                          images["Vegetation"],
                          images["Urban"],
                          images["Agricultural"]
                         ],
                         axis = 2
                        )

images["Combined"] = all_img

all_smooth_img = np.concatenate([smooth_images["Natural"], 
                                 smooth_images["Vegetation"],
                                 smooth_images["Urban"],
                                 smooth_images["Agricultural"]
                                ],
                                axis = 2
                               )

smooth_images["Combined"] = all_smooth_img

combo_results = cluster_ks(images["Combined"], range(3,10))
smooth_combo_results = cluster_ks(smooth_images["Combined"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = combo_results,
            smoothed_imgs = smooth_combo_results, 
            img_name = "Combined"
           )

dims = np.shape(images["Combined"])

print("image dimensions: {}".format(dims))
comb_img_matrix = np.reshape(images["Combined"], (dims[0] * dims[1], dims[2]))
print("flattened image dimensions: {}".format(np.shape(comb_img_matrix)))

corr = np.corrcoef(comb_img_matrix.T)
sns.heatmap(corr)

corr[:3].sum(axis = 0)

images["Natural_Agric"]  = np.concatenate([images["Natural"], 
                          images["Agricultural"]
                         ],
                         axis = 2
                        )


smooth_images["Natural_Agric"] = np.concatenate([smooth_images["Natural"],
                                     smooth_images["Agricultural"]
                                    ],
                                    axis = 2
                                   )

nat_agric_results = cluster_ks(images["Natural_Agric"], range(3,10))
nat_agric_combo_results = cluster_ks(smooth_images["Natural_Agric"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = nat_agric_results ,
            smoothed_imgs = nat_agric_combo_results, 
            img_name = "Natural_Agric"
           )

images["Natural_Veg"]  = np.concatenate([images["Natural"], 
                                         images["Vegetation"]
                                         ],
                                         axis = 2
                                       )


smooth_images["Natural_Veg"] = np.concatenate([smooth_images["Natural"],
                                                 smooth_images["Vegetation"]
                                              ],
                                              axis = 2
                                             )

nat_veg_results = cluster_ks(images["Natural_Veg"], range(3,10))
nat_veg_combo_results = cluster_ks(smooth_images["Natural_Veg"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = nat_veg_results,
            smoothed_imgs = nat_veg_combo_results, 
            img_name = "Natural_Veg"
           )

images["Natural_selective"] = images["Combined"][:,:, [1,2,3,4,9,10]]

smooth_images["Natural_selective"] = smooth_images["Combined"][:,:, [1,2,3,4,9,10]]

images["Natural_selective"].shape

smooth_images["Natural_selective"].shape

nat_sel_results = cluster_ks(images["Natural_selective"], range(3,10))
nat_sel_combo_results = cluster_ks(smooth_images["Natural_selective"], range(3,10))

plt_results(ks = range(3,10), 
            imgs = nat_sel_results,
            smoothed_imgs = nat_veg_combo_results, 
            img_name = "Natural_selective"
           )

