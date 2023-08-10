import torch
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import json
imagenet_classes = json.load(open('imagenet_classes.json'))
idx2class = [imagenet_classes[str(i)].split(',')[0] for i in range(1000)]
class2idx = {v:i for i,v in enumerate(idx2class)}

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
print(vgg16)

softmax = torch.nn.Softmax()

img_transforms = transforms.Compose([transforms.Scale((256, 256), Image.BICUBIC),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def norm(x):
    return 2.*(x/255.-0.5)

def unnorm(x):
    un_x = 255*(x*0.5+0.5)
    #un_x[un_x > 255] = 255
    #un_x[un_x < 0] = 0
    un_x = un_x.astype(np.uint8)
    return un_x

img = Image.open('espresso.png')
img_tensor = img_transforms(img)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(np.asarray(img))
plt.subplot(1,2,2)
plt.imshow(unnorm(img_tensor.numpy()).transpose(1,2,0))

img_var = torch.autograd.Variable(img_tensor.unsqueeze(0), requires_grad=True)
out = vgg16(img_var)
probs = softmax(out)
cls_idx = np.argmax(out.data.numpy())
print(str(cls_idx) + ":" + idx2class[cls_idx] + ":" + str(out.data.numpy()[0][cls_idx]) + ":" + str(probs.data.numpy()[0][cls_idx]))

out[0,class2idx['nematode']].backward()

grad_sign = np.sign(img_var.grad.data.numpy()).astype(np.uint8)
epsilon = 10
new_img_array = np.asarray(unnorm(img_tensor.numpy()))+epsilon*grad_sign
new_img_array[new_img_array>255] = 255
new_img_array[new_img_array<0] = 0
new_img_array = new_img_array.astype(np.uint8)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(unnorm(img_tensor.numpy()).transpose(1,2,0))
plt.subplot(1,2,2)
plt.imshow(new_img_array[0].transpose(1,2,0))
new_img_array = norm(new_img_array)
new_img_var = torch.autograd.Variable(torch.FloatTensor(new_img_array), requires_grad=True)
new_out = vgg16(new_img_var)
new_out_np = new_out.data.numpy()
new_probs = softmax(new_out)
new_cls_idx = np.argmax(new_out_np)
print(str(new_cls_idx) + ":" + idx2class[new_cls_idx] + ":" + str(new_probs.data.numpy()[0][new_cls_idx]))

learning_rate = 1
img = Image.open('espresso.png')
fake_img_tensor = img_transforms(img)
img_var_fake = torch.autograd.Variable(fake_img_tensor.unsqueeze(0), requires_grad=True)
fake_class_idx = class2idx['nematode']
for i in range(100):
    out_fake = vgg16(img_var_fake)
    _, out_idx = out_fake.data.max(dim=1)
    if out_idx.numpy() == fake_class_idx:
        print('Fake generated in ' + str(i) + ' iterations')
        break
    out_fake[0,fake_class_idx].backward()
    img_var_fake_grad = img_var_fake.grad.data
    img_var_fake.data += learning_rate*img_var_fake_grad/img_var_fake_grad.norm()
    img_var_fake.grad.data.zero_()
probs_fake = softmax(out_fake)
print(str(fake_class_idx) + ":" + idx2class[fake_class_idx] + ":" + str(out_fake.data.numpy()[0][fake_class_idx]) + ":" + str(probs_fake.data.numpy()[0][fake_class_idx]))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(unnorm(img_tensor.numpy()).transpose(1,2,0))
plt.subplot(1,2,2)
plt.imshow(unnorm(img_var_fake.data.numpy()[0]).transpose(1,2,0))



