import tensorflow as tf
tf.VERSION

import os

HOME_DIR = 'ubuntu'
DATA_DIR = os.path.join(HOME_DIR, 'data')
VOCAB_BIN = os.path.join(DATA_DIR, 'vocabulary.bin')
TRAIN_TFR = os.path.join(DATA_DIR, 'train.tfrecords')
VALID_TFR = os.path.join(DATA_DIR, 'valid.tfrecords')
TEST_TFR = os.path.join(DATA_DIR, 'test.tfrecords')

def has_file(file):
    if not os.path.isfile(file):
        raise Exception('File not found: {}'.format(file))

has_file(VOCAB_BIN)
has_file(TRAIN_TFR)
has_file(VALID_TFR)
has_file(TEST_TFR)

os.listdir(DATA_DIR)

# `tokenizer` function must be defined before restoring the vocabulary object
# (pickle does not serialize functions)
def tokenizer(sentences):
    return (sentence.split() for sentence in sentences)

class VocabularyAdapter:
    
    def __init__(self, vocabulary_bin):
        self._vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocabulary_bin)
    
    @property
    def size(self):
        return len(self._vocab.vocabulary_)
    
    @property
    def vector_length(self):
        return self._vocab.max_document_length

vocab = VocabularyAdapter(VOCAB_BIN)

print('Vocabulary size: {:,d}'.format(vocab.size))
print('Vector length: {:,d}'.format(vocab.vector_length))

with tf.Graph().as_default(), tf.Session() as session:
    filename_queue = tf.train.string_input_producer([TRAIN_TFR])

    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)

    example_features = {
        'context': tf.FixedLenFeature(vocab.vector_length, dtype=tf.int64),
        'context_len': tf.FixedLenFeature((), dtype=tf.int64),
        'utterance': tf.FixedLenFeature(vocab.vector_length, dtype=tf.int64),
        'utterance_len': tf.FixedLenFeature((), dtype=tf.int64),
        'label': tf.FixedLenFeature((), dtype=tf.int64),
    }

    example = tf.parse_single_example(value, example_features)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(3):
        example_ = session.run(example)
        context = example_['context']
        context_len = example_['context_len']
        utterance = example_['utterance']
        utterance_len = example_['utterance_len']
        label = example_['label']
        print('[ Example {} ]\n'.format(i))
        print('context\n\n{}\n'.format(context))
        print('context_len\n\n{}\n'.format(context_len))
        print('utterance\n\n{}\n'.format(utterance))
        print('utterance_len\n\n{}\n'.format(utterance_len))
        print('label\n\n{}\n'.format(label))

    coord.request_stop()
    coord.join(threads)

def features_train(vector_length):
    return [
        tf.feature_column.numeric_column(
            key='context', shape=vector_length, dtype=tf.int64),
        tf.feature_column.numeric_column(
            key='context_len', shape=1, dtype=tf.int64),
        tf.feature_column.numeric_column(
            key='utterance', shape=vector_length, dtype=tf.int64),
        tf.feature_column.numeric_column(
            key='utterance_len', shape=1, dtype=tf.int64),
        tf.feature_column.numeric_column(
            key='label', shape=1, dtype=tf.int64),
    ]

train_features = features_train(vocab.vector_length)

for x in train_features:
    print(x)

tf.feature_column.make_parse_example_spec(train_features)

def features_eval(vector_length):
    features = []
    keys = ['context', 'utterance']
    keys += ['distractor_{}'.format(i) for i in range(9)]
    for key in keys:
        features += [
            tf.feature_column.numeric_column(
                key=key, shape=vector_length, dtype=tf.int64),
            tf.feature_column.numeric_column(
                key=key + '_len', shape=1, dtype=tf.int64),
        ]
    return features

eval_features = features_eval(vocab.vector_length)

for x in eval_features:
    print(x)

tf.feature_column.make_parse_example_spec(eval_features)

with tf.Graph().as_default(), tf.Session() as session:
    example_features = tf.feature_column.make_parse_example_spec(train_features)
    
    batch_example = tf.contrib.learn.read_batch_record_features(
        file_pattern=[TRAIN_TFR],
        batch_size=3,
        features=example_features
    )
    
    print(batch_example, '\n')
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    examples = session.run(batch_example)
    
    for i in range(3):
        context = examples['context'][i]
        context_len = examples['context_len'][i]
        utterance = examples['utterance'][i]
        utterance_len = examples['utterance_len'][i]
        label = examples['label'][i]
        print('[ Example {} ]\n'.format(i))
        print('context\n\n{}\n'.format(context))
        print('context_len\n\n{}\n'.format(context_len))
        print('utterance\n\n{}\n'.format(utterance))
        print('utterance_len\n\n{}\n'.format(utterance_len))
        print('label\n\n{}\n'.format(label))

    coord.request_stop()
    coord.join(threads)

def _input_reader(name, filenames, features, batch_size, num_epochs):
    example_features = tf.feature_column.make_parse_example_spec(features)
    return tf.contrib.learn.read_batch_record_features(
        file_pattern=filenames,
        features=example_features,
        batch_size=batch_size,
        num_epochs=num_epochs,
        randomize_input=True,
        queue_capacity=200000 + batch_size * 10,
        name='read_batch_record_features_' + name
    )


def input_train(name, filenames, features, batch_size, num_epochs=None):
    batch_example = _input_reader(name, filenames, features, batch_size, num_epochs)
    batch_target = batch_example.pop('label')
    return batch_example, batch_target

def input_eval(name, filenames, features, batch_size, num_epochs=None):
    batch_example = _input_reader(name, filenames, features, batch_size, num_epochs)
    batch_target = tf.zeros_like(batch_example['context_len'])
    return batch_example, batch_target

input_fn_train = lambda: input_train('train', [TRAIN_TFR], train_features, 128, 1)
input_fn_valid = lambda: input_eval('valid', [VALID_TFR], eval_features, 16, 1)
input_fn_test = lambda: input_eval('test', [TEST_TFR], eval_features, 16, 1)

with tf.Graph().as_default(), tf.Session() as session:
    train_data = input_fn_train()
    valid_data = input_fn_valid()
    test_data = input_fn_test()
    
    tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    print('[ Training ]\n')
    
    batch_count = 0
    examples_count = 0

    try:
        while not coord.should_stop():
            data = session.run(train_data)
            examples_count += len(data[0]['context'])
            batch_count += 1
            if batch_count % 1000 == 0:
                print('... {:,d} examples'.format(examples_count))
    except tf.errors.OutOfRangeError:
        print('... {:,d} examples'.format(examples_count))
        print('Epoch limit reached')
    
    print()
    print('[ Validation ]\n')

    batch_count = 0
    examples_count = 0

    try:
        while not coord.should_stop():
            data = session.run(valid_data)
            examples_count += len(data[0]['context'])
            batch_count += 1
            if batch_count % 1000 == 0:
                print('... {:,d} examples'.format(examples_count))
    except tf.errors.OutOfRangeError:
        print('... {:,d} examples'.format(examples_count))
        print('Epoch limit reached')
    
    print()
    print('[ Test ]\n')

    batch_count = 0
    examples_count = 0

    try:
        while not coord.should_stop():
            data = session.run(test_data)
            examples_count += len(data[0]['context'])
            batch_count += 1
            if batch_count % 1000 == 0:
                print('... {:,d} examples'.format(examples_count))
    except tf.errors.OutOfRangeError:
        print('... {:,d} examples'.format(examples_count))
        print('Epoch limit reached')
    
    coord.request_stop()
    coord.join(threads)

