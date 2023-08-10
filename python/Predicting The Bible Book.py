import tensorflow as tf
from preppy import Preppy
class BibPreppy(Preppy):
    '''
    We'll slightly extend to way we right tfrecords to store the id of the book it came from
    '''
    def sequence_to_tf_example(self,sequence,book_id):
        id_list = self.sentance_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(sequence)
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        ex.context.feature["book_id"].int64_list.value.append(book_id)
        # Feature lists for the two sequential features of our example
        fl_tokens = ex.feature_lists.feature_list["tokens"]

        for token in id_list:
            fl_tokens.feature.add().int64_list.value.append(token)

        return ex
    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go froma  serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "book_id": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq":sequence_parsed["tokens"], "length": context_parsed["length"], 
                "book_id": context_parsed["book_id"]}


tf.reset_default_graph()

dataset = tf.data.TFRecordDataset(['./train.tfrecord']).map(BibPreppy.parse)

iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()

sess =tf.InteractiveSession()

sess.run(next_item)

dataset.output_shapes


def expand(x):
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']),0)
    x['book_id'] = tf.expand_dims(tf.convert_to_tensor(x['book_id']),0)
    return x
def deflate(x):
    x['length'] = tf.squeeze(x['length'])
    x['book_id'] = tf.squeeze(x['book_id'])
    return x


batch_iter = dataset.map(expand).padded_batch(128,padded_shapes={
    "book_id":1,
    "length":1,
    "seq":tf.TensorShape([None])
}).map(deflate)
next_item = batch_iter.repeat().make_one_shot_iterator().get_next()

sess.run(next_item)

class Model():
    def __init__(self,inputs):
        sequence =  inputs['seq']
        lengths = inputs['length']
        book_id = inputs['book_id']
        self.lr = tf.placeholder(shape=None,dtype=tf.float32)
        
        
        emb_vec = tf.get_variable("emb",dtype=tf.float32,shape=[74,32])
        emb_source = tf.nn.embedding_lookup(emb_vec,sequence)
        
        
        cell = tf.nn.rnn_cell.GRUCell(128)
        outputs, state = tf.nn.dynamic_rnn(cell,emb_source,dtype=tf.float32,sequence_length=lengths)
        
        book_logits =  tf.contrib.layers.fully_connected(state,num_outputs=64,activation_fn=tf.tanh)
        book_logits =  tf.contrib.layers.fully_connected(state,num_outputs=215,activation_fn=None)
        
        loss = tf.losses.sparse_softmax_cross_entropy(book_id,book_logits)
        self.loss = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(self.lr)
        self.train = opt.minimize(self.loss)


    

M = Model(next_item)
sess.run(tf.global_variables_initializer())
from IPython.display import clear_output

num =1
import sys
while True:
    try:
        _,loss = sess.run([M.train,M.loss],feed_dict={M.lr:0.0001})
        if num %30==0:
            clear_output()
        num+=1
        sys.stdout.write("\r" + str(loss))
        sys.stdout.flush()
    except:
        pass

    



