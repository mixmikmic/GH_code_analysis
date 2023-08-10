# The Model super class. It uses the specific models for calculation of the transformation
class Model(chainer.Chain):
    """
        This Model wrap other models like CDNA, STP or DNA.
        It calls their training and get the generated images and states, 
        it then compute the losses and other various parameters
    """
    
    def __init__(self, num_masks, is_cdna=True, is_dna=False, is_stp=False, prefix=None):
        """
            Initialize a CDNA, STP or DNA through this 'wrapper' Model
            Args:
                is_cdna: if the model should be an extension of CDNA
                is_dna: if the model should be an extension of DNA
                is_stp: if the model should be an extension of STP
                prefix: appended to the results to differentiate between training and validation
                learning_rate: learning rate
        """
        super(Model, self).__init__(
        enc0 = L.Convolution2D(in_channels=3, out_channels=32, ksize=(5, 5), stride=2, pad=5/2),
            enc1 = L.Convolution2D(in_channels=32, out_channels=32, ksize=(3,3), stride=2, pad=3/2),
            enc2 = L.Convolution2D(in_channels=64, out_channels=64, ksize=(3,3), stride=2, pad=3/2),
            enc3 = L.Convolution2D(in_channels=74, out_channels=64, ksize=(1,1), stride=1, pad=1/2),
            enc4 = L.Deconvolution2D(in_channels=128, out_channels=128, ksize=(3,3), 
                                     stride=2, outsize=(16,16), pad=3/2),
            enc5 = L.Deconvolution2D(in_channels=96, out_channels=96, ksize=(3,3), 
                                     stride=2, outsize=(32,32), pad=3/2),
            enc6 = L.Deconvolution2D(in_channels=64, out_channels=64, ksize=(3,3), 
                                     stride=2, outsize=(64,64), pad=3/2),

            lstm1 = BasicConvLSTMCell(in_size=None, out_size=32),
            lstm2 = BasicConvLSTMCell(in_size=None, out_size=32),
            lstm3 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm4 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm5 = BasicConvLSTMCell(in_size=None, out_size=128),
            lstm6 = BasicConvLSTMCell(in_size=None, out_size=64),
            lstm7 = BasicConvLSTMCell(in_size=None, out_size=32),
            
            norm_enc0 = LayerNormalizationConv2D(),
            norm_enc6 = LayerNormalizationConv2D(),
            hidden1 = LayerNormalizationConv2D(),
            hidden2 = LayerNormalizationConv2D(),
            hidden3 = LayerNormalizationConv2D(),
            hidden4 = LayerNormalizationConv2D(),
            hidden5 = LayerNormalizationConv2D(),
            hidden6 = LayerNormalizationConv2D(),
            hidden7 = LayerNormalizationConv2D(),

            masks = L.Deconvolution2D(in_channels=64, out_channels=num_masks+1, ksize=(1,1), stride=1),

            current_state = L.Linear(in_size=None, out_size=5)
    )
        self.num_masks = num_masks
        self.prefix = prefix

        model = None
        if is_cdna:
            model = StatelessCDNA(num_masks)
        elif is_stp:
            model = StatelessSTP(num_masks)
        elif is_dna:
            model = StatelessDNA(num_masks)
        if model is None:
            raise ValueError("No network specified")
        else:
            self.add_link('model', model)

    def __call__(self, images, actions=None, states=None, iter_num=-1.0, 
                 scheduled_sampling_k=-1, use_state=True, num_masks=10, num_frame_before_prediction=2):
         ### ...
        # Specific model transformations
        transformed = self.model(
            [lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7],
            [enc0, enc1, enc2, enc3, enc4, enc5, enc6],
            [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7],
            batch_size, prev_image, num_masks, int(color_channels)
        )
        #...

# CDNA, one of the three available model for pixel advection
class StatelessCDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using CDNA
        * Because the CDNA does not keep states, it should be passed as a parameter 
          if one wants to continue learning from previous states
    """
    
    def __init__(self, num_masks):
        super(StatelessCDNA, self).__init__(
            enc7 = L.Deconvolution2D(in_channels=64, out_channels=3, ksize=(1,1), stride=1),
            cdna_kerns = L.Linear(in_size=None, out_size=DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks)
        )

        self.num_masks = num_masks

    def __call__(self, lstm_states, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessCDNA.
            Args:
                lstm_states: An array of computed LSTM transformation
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)
        
        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5, lstm_state6, lstm_state7 = lstm_states
        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens
        
        #...
        
        return transformed
    
# Training loop
# ...
while train_iter.epoch < epoch:
    itr = train_iter.epoch
    batch = train_iter.next()
    img_training_set, act_training_set, sta_training_set = concat_examples(batch) # format the batch elements
    
    # Perform training
    logger.info("Begining training for mini-batch {0}/{1} of epoch {2}".format(
        str(train_iter.current_position), str(len(images_training)), str(itr+1))
    )
    optimizer.update(training_model, img_training_set, act_training_set, sta_training_set, 
                     itr, schedsamp_k, use_state, num_masks, context_frames)

