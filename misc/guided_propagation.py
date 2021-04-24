import torch
import torch.nn as nn


class GuidedBackprop():
    def __init__(self,model):
        self.model=model
        self.gradients=None
        self.forward_relu_outputs=[]
        self.model.cuda()
        self.model.eval()

    def hook_fn(self,module,grad_in,grad_out):
        self.gradients=grad_in[0]

    def hook_layers(self):
        # register hook to the first layer
        first_layer=list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(self.hook_fn)

    def update_relus(self):
        """ module for zeroing the gradients at backward pass """

        def relu_forward_hook(module,input,output):
            self.forward_relu_outputs.append(output)

        def relu_backward_hook(module,grad_in,grad_out):

            forward_output = self.forward_relu_outputs[-1]
            forward_output[forward_output>0] = 1
            modified_grads = forward_output * torch.clamp(grad_in,min=0.0)
            del self.forward_relu_outputs[-1]  # remove last relu forward output
            return (modified_grads,)

        # loop through layers abd hook up relus
        for pos,module in self.model.features._modules.items():
            if(isinstance(module,nn.Sequential)):
                if(isinstance(module,nn.ReLU)):
                    module.register_backward_hook(relu_backward_hook())
                    module.register_forward_hook(relu_forward_hook())
            elif(isinstance(module,nn.ReLU)):
                module.register_backward_hook(relu_backward_hook())
                module.register_forward_hook(relu_forward_hook())


    def get_gradients(self,input_img):
        pred = self.model(input_img.cuda())
        self.model.zero_grad()
        # target for backprop
        idx = pred.argmax(dim=1).item()
        pred[:,idx].backward()
        grads = self.gradients.detach().cpu().data.numpy()
        return grads


if __name__=="__main__":
    model = "trained_cnn_intf_free_vsg20"
    gbp = GuidedBackprop(model)
    guided_grads = gbp.get_gradients()