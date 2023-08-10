get_ipython().magic('matplotlib inline')

from torch.autograd import Variable
from fully_conv_net import FCResNet
from functions import load_image, build_heatmap, show_heatmap

def fully_conv(image_path, synset_id, mode="show", image_size=None):
    fc_net = FCResNet()
    fc_net.load_weight()

    input = load_image(image_path, image_size)
    # batch size = 1
    input = Variable(input).unsqueeze(0)
    # turn on eval mode not to use batch-normalization
    # dropout like training phase
    fc_net.eval()
    output = fc_net(input)
    hmap = build_heatmap(output.data, synset_id)
    show_heatmap(hmap, image_path, mode)

# synnets ID for dog and cat

dog_s = "n02084071"
cat_s = "n02121808"

fully_conv("images/cat.jpg", cat_s)

fully_conv("images/cat2.jpg", cat_s)

fully_conv("images/dog.jpg", dog_s)

fully_conv("images/dog2.jpg", dog_s)

fully_conv("images/dog.jpg", cat_s)



