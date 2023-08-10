# TODO: loss strategy
import tensorflow as tf

num_classes = 21
alpha = 1.0
neg_pos_ratio = 3.0
background_label_id = 0
negatives_for_hard = 100.0

# common offset calculation between y_true and y_pred
# smooth offset loss
def l1_smooth_loss(y_true, y_pred):
    abs_loss = tf.abs(y_true - y_pred)
    sq_loss = 0.5 * (y_true - y_pred)**2
    # get coordinates from sq_loss as row and (abs_loss - 0.5) as column
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    # calculate mean of each row
    return tf.reduce_mean(l1_loss, -1)

# softmax
def softmax_loss(y_true, y_pred):
    # given probability of prediction as [1e-15, 1 - 1e-15]
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
    # mean by row
    loss = - tf.reduce_mean(y_true * tf.log(y_pred), axis=-1)
    return loss

def compute_loss(y_true, y_pred):
    batch_size = tf.shape(y_true)[0]
    num_boxes = tf.to_float(tf.shape(y_true)[1])
    
    # 1. 
    # loss for all priors
    # class
    conf_loss = softmax_loss(y_true[:, :, 4:-8], y_pred[:, :, 4:-8])
    # coordinates of box
    loc_loss = l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])
    
    # 2.
    # get positives loss
    pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8], axis=1)
    pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8], axis=1)
    
    # mean all batches, shape is (batch, 1)
    num_pos = tf.reduce_mean(y_true[:, :, -8], axis=-1)
    
    # get negatives loss, we panalize only confidence
    # traverse each of batches to get num of negatives
    
    # 3.
    # get num of neg of batches
    num_neg_of_batches = tf.minimum(neg_pos_ratio * num_pos, num_boxes - num_pos)
    pos_num_neg_mask = tf.greater(num_neg_of_batches, 0)
    has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
    
    # [num_neg, negatives_for_hard (if True)]
    num_neg_of_batches = tf.concat(values=[num_neg_of_batches, [(1-has_min) * negatives_for_hard]], axis=0)
    # find min_num among batches as batch size of num_neg
    num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg_of_batches, tf.greater(num_neg_of_batches, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    
    # 4.
    # find max confidence probability
    confs_start = 4 + background_label_id + 1
    confs_end = confs_start + num_classes - 1
    max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end], axis=2)
    
    # get idx of top-k confidences 
    _, indices = tf.nn.top_k(max_confs * (1-y_true[:, :, -8]), k=num_neg_batch)
    # shape of (batch,1)
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    # negative idx
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) 
                    + tf.reshape(indices, [-1]))
    
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)
    
    # 5.
    # sum of
    total_loss = pos_conf_loss + neg_conf_loss
    total_loss /= (num_pos + tf.to_float(num_neg_batch))
    
    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
    
    total_loss += (alpha * pos_loc_loss) / num_pos
    return total_loss

