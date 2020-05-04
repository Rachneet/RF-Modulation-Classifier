import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as tf
import torch.nn as nn

# from visualize import SaveFeatures
import resnet


class SaveFeatures():
    def __init__(self, module, backward=False):
        if backward==False:
            if isinstance(module, nn.Sequential):
                print("in")
                for name,layer in module._modules.items():
                    self.hook = layer.register_forward_hook(self.hook_fn)
            else:
                self.hook = module.register_forward_hook(self.hook_fn)
        else:
            if isinstance(module, nn.Sequential):
                for name,layer in module._modules.items():

                    self.hook = layer.register_backward_hook(self.hook_fn_backward)
            else:
                self.hook = module.register_backward_hook(self.hook_fn_backward)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()

    def hook_fn_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out

    def close(self):
        self.hook.remove()


def grad_cam(path,input):

    # pass to pre trained model
    model = torch.load(path+"thesis_results/trained_resnet101_spectrogram_16k")
    model.eval()

    layers = dict(model.named_children())['encoder']
    print(type(layers))



    # get activations for the last convolution layer
    activations = SaveFeatures(list(model.children())[0])
    grads = SaveFeatures(list(model.children())[0], backward=True)

    pred = model(torch.Tensor(input).unsqueeze(dim=0).cuda())
    idx = pred.argmax(dim=1).item()
    # print(idx)
    pred[:,idx].backward()
    features = activations.features
    # print(features)
    # print(grads.gradients[0].size())
    pooled_gradients = torch.mean(grads.gradients[0], dim=[0, 2, 3])  # global avg pooled over the ht and wt dims
                                                                     # to obatin the neuron importance wts.
    # weight the channels by corresponding gradients
    # weighted combination of forward activation maps
    for i in range(64):
        features[:, i, :, :] *= pooled_gradients[i]

    # Note: this has to be followed by Relu acc to the paper
    # consider adding it
    # this will give us a heatmap of same dims as the last conv layer dims; 14x14 in case of VGG for example

    features = features.detach().cpu().data.numpy()
    features = features.flatten()
    features = np.reshape(features, (-1, 8, 8))
    features = features.transpose(1, 2, 0)

    # plt.imshow(features[:, :, 0])
    # plt.show()

    heatmap = features[:, :, 0]
    img = cv2.imread(path+"rf_dataset_inets/spectrogram_dataset/test/0/Fig_65331_mod_0.png")
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_AUTUMN)
    superimposed_img = heatmap * 0.4 + img
    # print(superimposed_img.shape)
    plt.imshow(superimposed_img[:,:,0])
    plt.show()
    cv2.imwrite('heatmap.jpg', superimposed_img)

    activations.close()



if __name__=="__main__":
    path = "/media/backup/Arsenal/"
    transform = tf.Compose(
        [tf.Resize(400), tf.ToTensor(),
         tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fig_path = path+"rf_dataset_inets/spectrogram_dataset/test/0/Fig_65331_mod_0.png"
    input = Image.open(fig_path)
    input = input.convert(mode='RGB')
    # input = np.array(input)
    # print(input.shape)
    input = transform(input)
    print(input.shape)
    grad_cam(path,input)