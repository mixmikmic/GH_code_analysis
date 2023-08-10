import cPickle as pickle
import visualize

# allows plots to show inline in ipython notebook
get_ipython().magic('matplotlib inline')

# Specify paths
FOLDER_PATH = 'loss_acc_values/'
filepath_list = ['googlenet_stats_out', 'vgg16_stats_out']

for filepath in filepath_list:
    full_filepath = FOLDER_PATH+filepath
    (train_loss, val_loss, val_acc, test_acc) = pickle.load(open(full_filepath, 'r'))
    data_set_name = filepath
    num_train = 140000
    ep = 200
    xlabel = 'epochs'
    print data_set_name
    visualize.plot_train_loss_val_loss_val_acc(filepath, train_loss, val_loss, val_acc, ep, num_train, xlabel)

FOLDER_PATH = 'loss_acc_values/'
filepath_list = ['highlr_5_googlenet_latlng.pkl', 'highlr_5_subset_vgg16_latlng.pkl']

train_loss_list = []
google_full_filepath = FOLDER_PATH+filepath_list[0]
google_train_loss = pickle.load(open(google_full_filepath, 'r'))
vgg_full_filepath = FOLDER_PATH+filepath_list[1]
vgg_train_loss = pickle.load(open(vgg_full_filepath, 'r'))

data_set_name = 'googlenet_vggnet_loss'
lr='1e-4'
ep = 500
xlabel = 'epochs'

legend_1 = 'GoogLeNet Losses'
legend_2 = 'VGG Net Losses'
print data_set_name

visualize.plot_loss(data_set_name, google_train_loss, vgg_train_loss, lr, ep, xlabel, legend_1, legend_2)   



