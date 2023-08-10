from __future__ import print_function
import glob
import os
import numpy as np
from PIL import Image
# Some of the flowers data is stored as .mat files
from scipy.io import loadmat
from shutil import copyfile
import sys
import tarfile
import time

# Loat the right urlretrieve based on python version
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
    
import zipfile

# Useful for being able to dump images into the Notebook
import IPython.display as D

# Import CNTK and helpers
import cntk
import cntk as C
# Load and convert data
from cntk.io import MinibatchSource, ImageDeserializer, StreamDefs, StreamDef
import cntk.io.transforms as xforms
from cntk import load_model, combine, softmax, Trainer, UnitType,  CloneMethod
from cntk.layers import Dense
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error

from cntk.logging import log_number_of_parameters, ProgressPrinter
# Interrogate the Compute Graph to find the right layer in the trained model
from cntk.logging.graph import find_by_name, get_node_outputs

isFast = True

# Check for an environment variable defined in CNTK's test infrastructure
def is_test(): return 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ

# Select the right target device when this notebook is being tested
# Currently supported only for GPU 
# Setup data environment for pre-built data sources for testing
if is_test(): 
    if 'TEST_DEVICE' in os.environ:
        if os.environ['TEST_DEVICE'] == 'cpu':
            raise ValueError('This notebook is currently not support on CPU') 
        else:
            cntk.device.try_set_default_device(cntk.device.gpu(0))
    sys.path.append(os.path.join(*"../Tests/EndToEndTests/CNTKv2Python/Examples".split("/")))
    import prepare_test_data as T
    T.prepare_resnet_v1_model()
    T.prepare_flower_data()
    T.prepare_animals_data() 

# By default, we store data in the Examples/Image directory under CNTK
# If you're running this _outside_ of CNTK, consider changing this
data_root = os.path.join('..', 'Examples', 'Image')
    
datasets_path = os.path.join(data_root, 'DataSets')
output_path = os.path.join('.', 'temp', 'Output')

def ensure_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(file_path, img_paths, img_labels):
    with open(file_path, 'w+') as f:
        for i in range(0, len(img_paths)):
            f.write('%s\t%s\n' % (os.path.abspath(img_paths[i]), img_labels[i]))

def download_unless_exists(url, filename, max_retries=3):
    '''Download the file unless it already exists, with retry. Throws if all retries fail.'''
    if os.path.exists(filename):
        print('Reusing locally cached: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        retry_cnt = 0
        while True:
            try:
                urlretrieve(url, filename)
                print('Download completed.')
                return
            except:
                retry_cnt += 1
                if retry_cnt == max_retries:
                    print('Exceeded maximum retry count, aborting.')
                    raise
                print('Failed to download, retrying.')
                time.sleep(np.random.randint(1,10))
        
def download_model(model_root = os.path.join(data_root, 'PretrainedModels')):
    ensure_exists(model_root)
    resnet18_model_uri = 'https://www.cntk.ai/Models/ResNet/ResNet_18.model'
    resnet18_model_local = os.path.join(model_root, 'ResNet_18.model')
    download_unless_exists(resnet18_model_uri, resnet18_model_local)
    return resnet18_model_local

def download_flowers_dataset(dataset_root = os.path.join(datasets_path, 'Flowers')):
    ensure_exists(dataset_root)
    flowers_uris = [
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
        'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
    ]
    flowers_files = [
        os.path.join(dataset_root, '102flowers.tgz'),
        os.path.join(dataset_root, 'imagelabels.mat'),
        os.path.join(dataset_root, 'setid.mat')
    ]
    for uri, file in zip(flowers_uris, flowers_files):
        download_unless_exists(uri, file)
    tar_dir = os.path.join(dataset_root, 'extracted')
    if not os.path.exists(tar_dir):
        print('Extracting {} to {}'.format(flowers_files[0], tar_dir))
        os.makedirs(tar_dir)
        tarfile.open(flowers_files[0]).extractall(path=tar_dir)
    else:
        print('{} already extracted to {}, using existing version'.format(flowers_files[0], tar_dir))

    flowers_data = {
        'data_folder': dataset_root,
        'training_map': os.path.join(dataset_root, '6k_img_map.txt'),
        'testing_map': os.path.join(dataset_root, '1k_img_map.txt'),
        'validation_map': os.path.join(dataset_root, 'val_map.txt')
    }
    
    if not os.path.exists(flowers_data['training_map']):
        print('Writing map files ...')
        # get image paths and 0-based image labels
        image_paths = np.array(sorted(glob.glob(os.path.join(tar_dir, 'jpg', '*.jpg'))))
        image_labels = loadmat(flowers_files[1])['labels'][0]
        image_labels -= 1

        # read set information from .mat file
        setid = loadmat(flowers_files[2])
        idx_train = setid['trnid'][0] - 1
        idx_test = setid['tstid'][0] - 1
        idx_val = setid['valid'][0] - 1

        # Confusingly the training set contains 1k images and the test set contains 6k images
        # We swap them, because we want to train on more data
        write_to_file(flowers_data['training_map'], image_paths[idx_train], image_labels[idx_train])
        write_to_file(flowers_data['testing_map'], image_paths[idx_test], image_labels[idx_test])
        write_to_file(flowers_data['validation_map'], image_paths[idx_val], image_labels[idx_val])
        print('Map files written, dataset download and unpack completed.')
    else:
        print('Using cached map files.')
        
    return flowers_data
    
def download_animals_dataset(dataset_root = os.path.join(datasets_path, 'Animals')):
    ensure_exists(dataset_root)
    animals_uri = 'https://www.cntk.ai/DataSets/Animals/Animals.zip'
    animals_file = os.path.join(dataset_root, 'Animals.zip')
    download_unless_exists(animals_uri, animals_file)
    if not os.path.exists(os.path.join(dataset_root, 'Test')):
        with zipfile.ZipFile(animals_file) as animals_zip:
            print('Extracting {} to {}'.format(animals_file, dataset_root))
            animals_zip.extractall(path=os.path.join(dataset_root, '..'))
            print('Extraction completed.')
    else:
        print('Reusing previously extracted Animals data.')
        
    return {
        'training_folder': os.path.join(dataset_root, 'Train'),
        'testing_folder': os.path.join(dataset_root, 'Test')
    }

print('Downloading flowers and animals data-set, this might take a while...')
flowers_data = download_flowers_dataset()
animals_data = download_animals_dataset()
print('All data now available to the notebook!')

print('Downloading pre-trained model. Note: this might take a while...')
base_model_file = download_model()
print('Downloading pre-trained model complete!')

# define base model location and characteristics
base_model = {
    'model_file': base_model_file,
    'feature_node_name': 'features',
    'last_hidden_node_name': 'z.x',
    # Channel Depth x Height x Width
    'image_dims': (3, 224, 224)
}

# Print out all layers in the model
print('Loading {} and printing all layers:'.format(base_model['model_file']))
node_outputs = get_node_outputs(load_model(base_model['model_file']))
for l in node_outputs: print("  {0} {1}".format(l.name, l.shape))

def plot_images(images, subplot_shape):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(*subplot_shape)
    for image, ax in zip(images, axes.flatten()):
        ax.imshow(image.reshape(28, 28), vmin = 0, vmax = 1.0, cmap = 'gray')
        ax.axis('off')
    plt.show()

flowers_image_dir = os.path.join(flowers_data['data_folder'], 'extracted', 'jpg')


for image in ['08093', '08084', '08081', '08058']:
    D.display(D.Image(os.path.join(flowers_image_dir, 'image_{}.jpg'.format(image)), width=100, height=100))

ensure_exists(output_path)
np.random.seed(123)

# Creates a minibatch source for training or testing
def create_mb_source(map_file, image_dims, num_classes, randomize=True):
    transforms = [xforms.scale(width=image_dims[2], height=image_dims[1], channels=image_dims[0], interpolations='linear')]
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
            features =StreamDef(field='image', transforms=transforms),
            labels   =StreamDef(field='label', shape=num_classes))),
            randomize=randomize)

# Creates the network model for transfer learning
def create_model(model_details, num_classes, input_features, new_prediction_node_name='prediction', freeze=False):
    # Load the pretrained classification net and find nodes
    base_model   = load_model(model_details['model_file'])
    feature_node = find_by_name(base_model, model_details['feature_node_name'])
    last_node    = find_by_name(base_model, model_details['last_hidden_node_name'])

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: C.placeholder(name='features')})

    # Add new dense layer for class prediction
    feat_norm  = input_features - C.Constant(114)
    cloned_out = cloned_layers(feat_norm)
    z          = Dense(num_classes, activation=None, name=new_prediction_node_name) (cloned_out)

    return z

# Trains a transfer learning model
def train_model(model_details, num_classes, train_map_file,
                learning_params, max_images=-1):
    num_epochs = learning_params['max_epochs']
    epoch_size = sum(1 for line in open(train_map_file))
    if max_images > 0:
        epoch_size = min(epoch_size, max_images)
    minibatch_size = learning_params['mb_size']
    
    # Create the minibatch source and input variables
    minibatch_source = create_mb_source(train_map_file, model_details['image_dims'], num_classes)
    image_input = C.input(model_details['image_dims'])
    label_input = C.input(num_classes)

    # Define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source['features'],
        label_input: minibatch_source['labels']
    }

    # Instantiate the transfer learning model and loss function
    tl_model = create_model(model_details, num_classes, image_input, freeze=learning_params['freeze_weights'])
    ce = cross_entropy_with_softmax(tl_model, label_input)
    pe = classification_error(tl_model, label_input)

    # Instantiate the trainer object
    lr_schedule = learning_rate_schedule(learning_params['lr_per_mb'], unit=UnitType.minibatch)
    mm_schedule = momentum_schedule(learning_params['momentum_per_mb'])
    learner = momentum_sgd(tl_model.parameters, lr_schedule, mm_schedule, 
                           l2_regularization_weight=learning_params['l2_reg_weight'])
    trainer = Trainer(tl_model, (ce, pe), learner)

    # Get minibatches of images and perform model training
    print("Training transfer learning model for {0} epochs (epoch_size = {1}).".format(num_epochs, epoch_size))
    log_number_of_parameters(tl_model)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=num_epochs)
    for epoch in range(num_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = minibatch_source.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % (100 * minibatch_size) == 0:
                print ("Processed {0} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)

    return tl_model

# Evaluates a single image using the re-trained model
def eval_single_image(loaded_model, image_path, image_dims):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    try:
        img = Image.open(image_path)
        
        if image_path.endswith("png"):
            temp = Image.new("RGB", img.size, (255, 255, 255))
            temp.paste(img, img)
            img = temp
        resized = img.resize((image_dims[2], image_dims[1]), Image.ANTIALIAS)
        bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {loaded_model.arguments[0]: [hwc_format]}
        output = loaded_model.eval(arguments)

        # return softmax probabilities
        sm = softmax(output[0])
        return sm.eval()
    except FileNotFoundError:
        print("Could not open (skipping file): ", image_path)
        return ['None']
        


# Evaluates an image set using the provided model
def eval_test_images(loaded_model, output_file, test_map_file, image_dims, max_images=-1, column_offset=0):
    num_images = sum(1 for line in open(test_map_file))
    if max_images > 0:
        num_images = min(num_images, max_images)
    if isFast:
        num_images = min(num_images, 300) #We will run through fewer images for test run
        
    print("Evaluating model output node '{0}' for {1} images.".format('prediction', num_images))

    pred_count = 0
    correct_count = 0
    np.seterr(over='raise')
    with open(output_file, 'wb') as results_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0 + column_offset]
                probs = eval_single_image(loaded_model, img_file, image_dims)
                
                if probs[0]=='None':
                    print("Eval not possible: ", img_file)
                    continue

                pred_count += 1
                true_label = int(tokens[1 + column_offset])
                predicted_label = np.argmax(probs)
                if predicted_label == true_label:
                    correct_count += 1

                #np.savetxt(results_file, probs[np.newaxis], fmt="%.3f")
                if pred_count % 100 == 0:
                    print("Processed {0} samples ({1:.2%} correct)".format(pred_count, 
                                                                           (float(correct_count) / pred_count)))
                if pred_count >= num_images:
                    break
    print ("{0} of {1} prediction were correct".format(correct_count, pred_count))
    return correct_count, pred_count, (float(correct_count) / pred_count)

force_retraining = True

max_training_epochs = 5 if isFast else 20
print(flowers_data["training_map"])
learning_params = {
    'max_epochs': max_training_epochs,
    'mb_size': 50,
    'lr_per_mb': [0.2]*10 + [0.1],
    'momentum_per_mb': 0.9,
    'l2_reg_weight': 0.0005,
    'freeze_weights': True
}

flowers_model = {
    'model_file': os.path.join(output_path, 'FlowersTransferLearning.model'),
    'results_file': os.path.join(output_path, 'FlowersPredictions.txt'),
    'num_classes': 102
}

# Train only if no model exists yet or if force_retraining is set to True
if os.path.exists(flowers_model['model_file']) and not force_retraining:
    print("Loading existing model from %s" % flowers_model['model_file'])
    trained_model = load_model(flowers_model['model_file'])
else:
    trained_model = train_model(base_model,
                                flowers_model['num_classes'], flowers_data['training_map'],
                                learning_params)
    trained_model.save(flowers_model['model_file'])
    print("Stored trained model at %s" % flowers_model['model_file'])
    

print(flowers_model["results_file"])
# Evaluate the test set
predict_correct, predict_total, predict_accuracy =    eval_test_images(trained_model, flowers_model['results_file'], flowers_data['testing_map'], base_model['image_dims'])
print("Done. Wrote output to %s" % flowers_model['results_file'])

# Test: Accuracy on flower data
print ("Prediction accuracy: {0:.2%}".format(float(predict_correct) / predict_total))

sheep = ['738519_d0394de9.jpg', 'Pair_of_Icelandic_Sheep.jpg']
wolves = ['European_grey_wolf_in_Prague_zoo.jpg', 'Wolf_je1-3.jpg']
for image in [os.path.join('Sheep', f) for f in sheep] + [os.path.join('Wolf', f) for f in wolves]:
    D.display(D.Image(os.path.join(animals_data['training_folder'], image), width=100, height=100))

# Set python version variable 
python_version = sys.version_info.major

def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False, valid_extensions=['.jpg', '.jpeg', '.png']):
    map_file_name = os.path.join("", "map.txt")
    
    map_file = None
    #map_file_name = map_file_name.replace("Examples", "Examples2")
    if python_version == 3: 
        map_file = open(map_file_name , 'w', encoding='utf-8')
    else:
        map_file = open(map_file_name , 'w')

    for class_id in range(0, len(class_mapping)):
        folder = os.path.join(os.path.realpath(root_folder), class_mapping[class_id])
        if os.path.exists(folder):
            for entry in os.listdir(folder):
                filename = os.path.abspath(os.path.join(folder, entry))
                if os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in valid_extensions:
                    try:
                        map_file.write("{0}\t{1}\n".format(filename, class_id))
                    except UnicodeEncodeError:
                        continue

    if include_unknown:
        for entry in os.listdir(root_folder):
            filename = os.path.abspath(os.path.join(os.path.realpath(root_folder), entry))
            if os.path.isfile(filename) and os.path.splitext(filename)[1].lower() in valid_extensions:
                try:
                    map_file.write("{0}\t-1\n".format(filename))
                except UnicodeEncodeError:
                    continue
                    
    map_file.close()  
    
    return map_file_name
  

def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    return np.asarray(classes)

animals_data['class_mapping'] = create_class_mapping_from_folder(animals_data['training_folder'])
animals_data['training_map'] = create_map_file_from_folder(animals_data['training_folder'] , animals_data['class_mapping'])
# Since the test data includes some birds, set include_unknown
animals_data['testing_map'] = create_map_file_from_folder(animals_data['testing_folder'], animals_data['class_mapping'], 
                                                          include_unknown=False)

animals_data["testing_folder"]

animals_model = {
    'model_file': os.path.join(output_path, 'AnimalsTransferLearning.model'),
    'results_file': os.path.join(output_path, 'AnimalsPredictions.txt'),
    'num_classes': len(animals_data['class_mapping'])
}

force_retraining = False
if os.path.exists(animals_model['model_file']) and not force_retraining:
    print("Loading existing model from %s" % animals_model['model_file'])
    trained_model = load_model(animals_model['model_file'])
else:
    print("Building Model")
    trained_model = train_model(base_model, 
                                animals_model['num_classes'], animals_data['training_map'],
                                learning_params)
    trained_model.save(animals_model['model_file'])
    print("Stored trained model at %s" % animals_model['model_file'])

animals_model["model_file"]

# evaluate test images
with open(animals_data['testing_map'], 'r') as input_file:
    for line in input_file:
        tokens = line.rstrip().split('\t')
        img_file = tokens[0]
        true_label = int(tokens[1])
        probs = eval_single_image(trained_model, img_file, base_model['image_dims'])

        if probs[0]=='None':
            continue
        class_probs = np.column_stack((probs, animals_data['class_mapping'])).tolist()
        class_probs.sort(key=lambda x: float(x[0]), reverse=True)
        predictions = ' '.join(['%s:%.3f' % (class_probs[i][1], float(class_probs[i][0]))                                 for i in range(0, animals_model['num_classes'])])
        true_class_name = animals_data['class_mapping'][true_label] if true_label >= 0 else 'unknown'
        print('Class: %s, predictions: %s, image: %s' % (true_class_name, predictions, img_file))

images = ['Bird_in_flight_wings_spread.jpg', 'quetzal-bird.jpg', 'Weaver_bird.jpg']
for image in images:
    D.display(D.Image(os.path.join(animals_data['testing_folder'], image), width=100, height=100))



