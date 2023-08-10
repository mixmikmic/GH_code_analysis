import numpy as np
import tensorflow as tf

# Load the dataset into a numpy keyed structure
newsgroups = np.load('./resources/newsgroup.npz')

# Define the batch size
batch_size = 100

def dataset_input_fn(dataset):
    """
    Creates an input function using the `numpy_input_fn` method from
    tensorflow, based on the dataset we want to use.
    
    Args:
        dataset: String that represents the dataset
        (should be `train` or `test`)
    
    Returns:
        An `numpy_input_fn` function to feed to an estimator
    """
    assert dataset in ('train', 'test'),        "The selected dataset should be `train` or `test`"
    
    return tf.estimator.inputs.numpy_input_fn(
        # A dictionary of numpy arrays that match each array with the
        # corresponding column in the model. For this case we only
        # have "one" colum which represents the whole array.
        x={'input_data': newsgroups['%s_data' % dataset]},
        # The target array
        y=newsgroups['%s_target' % dataset],
        # The batch size to iterate the data in small fractions
        batch_size=batch_size,
        # If the dataset is `test` only run once
        num_epochs=1 if dataset == 'test' else None,
        # Only shuffle the dataset for the `train` data
        shuffle=dataset == 'train'
    )

input_size = newsgroups['train_data'].shape[1]
num_classes = newsgroups['labels'].shape[0]

feature_columns = [tf.feature_column.numeric_column(
    'input_data', shape=(input_size,))]

big_model = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=(5000, 2000,),
    n_classes=num_classes,
    model_dir="models/bigDNN")

big_experiment = tf.contrib.learn.Experiment(
    big_model,
    train_input_fn=dataset_input_fn("train"),
    eval_input_fn=dataset_input_fn("test"),
    train_steps=2000,
    train_steps_per_iteration=100
)
big_experiment.continuous_train_and_eval()

little_model = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=(500, 100,),
    n_classes=num_classes,
    model_dir="models/littleDNN")

little_experiment = tf.contrib.learn.Experiment(
    little_model,
    train_input_fn=dataset_input_fn("train"),
    eval_input_fn=dataset_input_fn("test"),
    train_steps=2000,
    train_steps_per_iteration=100
)
little_experiment.continuous_train_and_eval()

single_layer_model = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=(1000,),
    n_classes=num_classes,
    model_dir="models/single_layer_NN")

single_layer_experiment = tf.contrib.learn.Experiment(
    single_layer_model,
    train_input_fn=dataset_input_fn("train"),
    eval_input_fn=dataset_input_fn("test"),
    train_steps=2000,
    train_steps_per_iteration=100
)
single_layer_experiment.continuous_train_and_eval()



