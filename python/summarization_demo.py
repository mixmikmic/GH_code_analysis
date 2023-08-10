### Download the trained weights of the models ###
import os
if not os.path.exists('data/vgg19.pkl'):
    get_ipython().system('wget -P ./data/ https://data.vision.ee.ethz.ch/arunv/qv_summary_codes/vgg19.tar.gz -nc')
    get_ipython().system('tar -xzvf ./data/vgg19.tar.gz -C ./data/ --skip-old-files')
    get_ipython().system('rm ./data/vgg19.tar.gz')
get_ipython().system('wget -P ./data/ https://data.vision.ee.ethz.ch/arunv/qv_summary_codes/CNNmodel.npz -nc')
get_ipython().system('wget -P ./data/ https://data.vision.ee.ethz.ch/arunv/qv_summary_codes/LSTMmodel.npz -nc')
'''
Warning: The model below is of 10G. This is complete version. May take some time to download and unzip. 
For a smaller model, comment this and use the commented model further below.
'''
get_ipython().system('wget -P ./data/ https://data.vision.ee.ethz.ch/arunv/qv_summary_codes/word2vecmodel.tar.gz -nc')
get_ipython().system('mkdir ./data/word2vec')
get_ipython().system('tar -xzvf ./data/word2vecmodel.tar.gz -C ./data/word2vec/ --skip-old-files')

## Small model, Use below one: This has a memory of 3.5G ##

#!wget -P ./data/word2vec/ https://data.vision.ee.ethz.ch/arunv/qv_summary_codes/GoogleNews-vectors-negative300.bin -nc

### Import our package ###
import qvsumm

'''
Set and compile the relevance score function
On the GPU, the network will be using cuDNN layer implementations available in the Lasagne.

'''
relscore_function = qvsumm.get_QAR_function()

### Load the word2vec model ###
w2vmodel = qvsumm.get_word2vec_function()

# Inputs : Text query and YouTube URL of the video
videoURL="https://www.youtube.com/watch?v=ntza_9lbbc0"
query="cat fails"

# Preprocessing Data 
from qvsumm.utils_func import preprocess_video
imagenames= preprocess_video(query,videoURL)

### Compute the similarity and quality scores for all the frames of the video ###
relscores,intscores=qvsumm.get_rel_Q_scores(relscore_function, w2vmodel, query, imagenames)  

'''
Summarization using submodular mixtures[1] having four objective functions: Interestingness, Relevance, Diversity, Representativeness

[1] Gygli, Grabner & Van Gool. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015.
''' 
from qvsumm.shells import *
K=5 ### Number of elements in the summary
S1=qvsumm.shells.Summ(query,imagenames,relscores,intscores)
S1.budget=K
objectives=[quality_shell(S1),similarity_shell(S1),diversity_shell(S1),ex.representativeness_shell(S1)]
weights=[0.00963344,  0.45267703,  0.43680236,  0.10088718 ]###weights for the objectives
selected_elements,score,_=gm_submodular.lazy_greedy_maximize(S1,weights,objectives,budget=K,randomize=True)

print selected_elements

'''
We extract the 5 frames from the video.
'''
import matplotlib .pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
plt.figure(figsize=(60, 10))
for enum,i in enumerate(selected_elements[0:K]):
    if enum==3:
        plt.title("Query: "+str(query),fontsize=70)
    plt.subplot(1,K, enum+1);plt.imshow(mpimg.imread("videos/frames/"+str(i)+".png"));plt.axis('off')
    plt.annotate(str(relscores[i]), xy=(1, 0), xycoords='axes fraction', fontsize=60,
    horizontalalignment='right', verticalalignment='bottom',color='green')
plt.show()



