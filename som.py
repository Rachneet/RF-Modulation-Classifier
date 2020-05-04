import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import dataloader as dl
from pytorch_model_summary import summary
import time
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
from minisom import MiniSom
from sklearn import preprocessing
import h5py as h5

import plotly.io as pio
pio.renderers.default = 'svg'
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly
from shutil import copyfile
import plotly.figure_factory as ff
import plotly_express as px

plt.interactive(True)
# print(plotly.__version__)

class SOM(nn.Module):
    # output size is m,n
    def __init__(self,m,n,in_dim,niter,alpha=None,sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.niter = niter
        self.dim = in_dim

        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)

        if sigma is None:
            self.sigma = max(m,n)/2.0
        else:
            self.sigma = float(sigma)

        # x,y = in_dim
        # w = torch.empty(in_dim,m*n)
        self.weights = nn.Parameter(torch.randn(in_dim,m*n),requires_grad=False)
        # print("weight shape",self.weights.shape)
        self.locations = nn.Parameter(torch.LongTensor(list(self.neuron_locations())),requires_grad=False)
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield (i,j)

    def map_input(self,input_vects):
        vects = []
        self.weights = nn.Parameter(self.weights.permute(1,0))
        for vect in input_vects:
            # print("vect shape:{}".format(vect.shape))
            # print("wts shape:{}".format(weights.shape))
            min_index = min([i for i in range(len(self.weights))],
                            key= lambda z: np.linalg.norm(vect-self.weights[z].detach().cpu()))
            vects.append(self.locations[min_index])
        return vects

    # gaussian neighbourhood function
    # defines how the weight will change
    def neighbourhood_fn(self,input,sigma):
        '''e^(-(input / sigma^2))'''
        input.div_(sigma**2)
        input.neg_()
        input.exp_()

        return input

    def batch_pairwise_squared_distances(self,x,y):
        '''
        Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
        Input: x is a bxNxd matrix y is an optional bxMxd matirx
        Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
        i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        x_norm = (x ** 2).sum(2).view(x.shape[0], x.shape[1], 1)
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = (y ** 2).sum(2).view(y.shape[0], 1, y.shape[1])
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        dist[dist != dist] = 0  # replace nan values with 0
        return torch.clamp(dist, 0.0, np.inf)

    def forward(self,input,current_iter):

        # input = input.view(-1,2)
        # input = input.permute(0,2,1)
        # print("in: ", input)
        batch_size = input.size()[0]
        # print("in_shp before view: ", input.shape)
        input = input.view(batch_size,-1,1)
        # input = input.unsqueeze(dim=3)

        # print("in_shp after view: ",input.shape)
        # adapt weights for batch
        batch_weight = self.weights.expand(batch_size,-1,-1)
        # print("batch weight shape: ",batch_weight.shape)

        # find location of bmu units
        dists = self.pdist(input,batch_weight)
        # dists[torch.isinf(dists)] = 0
        # dists[torch.isnan(dists)] = 0
        # print(dists)
        # print("distance shape: {}".format(dists.shape))
        loss,bmu_index = dists.min(dim=1,keepdim=True)  # get index of BMU with least distance from input

        bmu_loc = self.locations[bmu_index]
        # print("loss: {}".format(loss))
        # print("len loss: {}".format(len(loss)))
        # print("bmu shape: {}".format(bmu_loc.shape))

        # setting lr
        iter_correction = 1.0 - (current_iter / self.niter)
        lr = self.alpha * iter_correction
        sigma = self.sigma * iter_correction
        # print("location shape before dist: {}".format(self.locations.shape))

        distance_squares = self.locations.float() - bmu_loc.float()
        # print("dist sq shape: {}".format(distance_squares.shape))
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)
        # print("dist sq shape: {}".format(distance_squares.shape))

        lr_locations = self.neighbourhood_fn(distance_squares, sigma)
        # print("location shape: {}".format(lr_locations.shape))
        lr_locations.mul_(lr).unsqueeze_(1)
        # print("location shape2: {}".format(lr_locations.shape))
        # print("in shape: {}".format(input.shape))
        # print("wt shape: {}".format(self.weights.shape))
        # print((input - self.weights).shape)
        delta = lr_locations * (input - self.weights)
        delta = delta.sum(dim=0)
        delta.div_(batch_size)
        self.weights.data.add_(delta)
        # print(self.weights.shape)

        return loss.sum().div_(batch_size).item()


    def save_result(self, dir, im_size=(0, 0)):
        '''
        Visualizes the weight of the Self Oranizing Map(SOM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        # print("wt shape in saving: {}".format(self.weights.shape))
        images = self.weights.view(im_size[0], im_size[1], self.m * self.n)

        images = images.permute(2, 0, 1)
        save_image(images, dir, normalize=True, padding=1, nrow=self.m)



def save_plot(figure, plot_name):
    img_name = plot_name
    dload = os.path.expanduser('~/Downloads')
    save_dir = '/home/rachneet/PycharmProjects'


    # data = [go.Scatter(x=[1, 2, 3], y=[3, 2, 6])]

    plotly.offline.plot(figure, image_filename=img_name, image='svg')

    ### might need to wait for plot to download before copying
    time.sleep(5)

    copyfile('{}/{}.svg'.format(dload, img_name),
             '{}/{}.svg'.format(save_dir, img_name))


if __name__=="__main__":

    data_path = "/media/backup/Arsenal/rf_dataset_inets/"
    # x_train_gen, y_train_gen, x_val_gen, y_val_gen,data_raw,labels_raw = dl.load_batch(
    #     data_path + "test_sample.npz",batch_size=256,mode='train')
    f = h5.File(data_path+"feature_set_conv6_vsg20.h5", 'r')
    features, t_labels, pred_labels = f['features'], f['true_labels'], f['pred_labels']

    # x_train_gen = DataLoader(features[:61440],batch_size=256,shuffle=False)
    # print("Num Batches: {}".format(len(x_train_gen)))
    print("Data loaded and batched...")

    # features = list()
    # labels_ = list()
    # labels = list()
    # for _,l in enumerate(labels_raw):
    #     # print(x)
    #     labels_.append(dl.label_idx(l))

    # for i in range(len(labels_)):
    #     labels.append(torch.Tensor(labels_[i]))

    # for _, x in enumerate(data_raw):
    #     # print(x)
    #     y = x.flatten()
    #     features.append(y)

    data = list()
    for i in range(len(features)):
        data.append(torch.FloatTensor(features[i]))

    m,n,in_dim = 32,32,128
    num_epochs = 10000
    # epoch = 10000
    model = SOM(m,n,in_dim,num_epochs)
    model.load_state_dict(torch.load("trained_som_conv6_features.pth"))
    model.cuda()

    # input = torch.randn(1,1024,2)
    # input = input.view(-1,2*1024)
    # for iter in range(num_epochs):
    #     loss = model.forward(input.cuda(),iter,num_epochs)
    #     print(loss)
    #
    # print(summary(som,input))
    # losses = list()
    # for epoch in range(num_epochs):
    #     running_loss = 0
    #     start_time = time.time()
    #     for iter,batch in enumerate(x_train_gen):
    #
    #         # batch = batch.view(-1,2048).cuda()
    #         # batch = batch.permute(0,2,1).cuda()
    #         # batch = batch.reshape(-1,*batch.shape[0:1])
    #         # batch = batch.permute(1,0)
    #         # print(batch.shape)
    #         batch = batch.cuda()
    #
    #         loss = model.forward(batch,iter)
    #         # print(loss)
    #         running_loss += loss
    #
    #
    #     losses.append(running_loss)
    #     print('epoch = %d, loss = %.2f, time = %.2fs' % (epoch + 1, running_loss/len(x_train_gen), time.time() - start_time))
    #
    #
    #     if (num_epochs>0 and num_epochs%9==0):
    #         model.alpha = model.alpha/10
    #
    # torch.save(model.state_dict(), 'trained_som_conv6_features.pth')

        # if epoch%500==0 :

    # Store a centroid grid for easy retrieval later on
    centroid_grid = [[] for i in range(m)]
    weights = model.get_weights()
    # print(weights.shape)
    weights = weights.permute(1,0)
    locations = model.get_locations()
    # print(locations.shape)
    for i, loc in enumerate(locations):
        centroid_grid[loc[0]].append(weights[i].cpu().data.numpy())
    # print(len(centroid_grid[0][0]))
    # print(centroid_grid[0][0])

    # Get output grid
    image_grid = centroid_grid
    # print(len(image_grid))
    x = np.array(image_grid)
    result = x[:, :, 0]
    fig = go.Figure(go.Surface(z=result))

    # Map colours to their closest neurons
    num_points = 5000
    mapped = model.map_input(data[:num_points])

    mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                   "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
    # for 5 class
    mod_schemes_ = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                   "OFDM", "OFDM", "OFDM", "OFDM"]

    mods = t_labels[:num_points]
    # Plot
    mapper = list()
    for i, m in enumerate(mapped):
        mapper.append(m.detach().cpu().data.numpy())
    mapper = np.array(mapper)

    # print(mapper)
    # unique, counts = np.unique(mods, return_counts=True)
    # print(np.asarray((unique, counts)).T)

    trace_list = list()
    check_mod = list()
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#bcbd22',  # curry yellow-green
        ]
    # '#17becf'  # blue-teal

    colors_ = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#8c564b',  # muted purple
        '#8c564b',  # chestnut brown
        '#8c564b',  # raspberry yogurt pink
        '#8c564b',  # curry yellow-green
    ]


    for i,m in enumerate(mapper):
        if mods[i] not in check_mod:
            check_mod.append(mods[i])
            legend = True
        else:
            legend = False
        trace = go.Scatter3d(
            x=np.array(m[1]),
            y=np.array(m[0]),
            z=np.array([random.randint(0,31)]),
            name=mod_schemes[mods[i]],
            mode="markers",
                marker=dict(
                    size=4,
                    color=colors[mods[i]],
                    opacity=0.6,
                    colorscale="Viridis"
                ),
            showlegend=legend
        )

        # print(trace.name)
        trace_list.append(trace)
    ordered_trace=[]
    for mod in mod_schemes:
        for trace in trace_list:
            if trace.name==mod:
                ordered_trace.append(trace)

    # fig = px.imshow(result) # color_continuous_scale='gray')
    # all_traces=[]
    # all_traces.append(trace1)
    for trace in ordered_trace:
        fig.add_trace(trace)
    # print(all_traces)
    # fig = go.Figure(data=all_traces)

    # fig['layout'].update(title='Modulation Clustering')
    cbar_text = [0,0.5,1,1.5,2,2.5,3]
    # cbar_vals = list(range(len(result)))
    fig['data'][0].colorbar = dict(title='Distance',
                                   outlinecolor="black",
                                   outlinewidth=1,
                                   ticks='outside',
                                   x=0.9,
                                   tickvals=cbar_text,
                                   ticktext=cbar_text
                                   )

    fig['layout'].update(title=go.layout.Title(
        text="Modulation Clustering",
        xref="paper",
        font=dict(
            # family="sans-serif",
            size=20,
            color="black"
        )
    )
    )


    width=500
    l_0, l_1 = 32,32

    fig.update_layout(scene=dict(yaxis=dict(zeroline=False,showticklabels=False,
                                 ticks="",showgrid=True,title='',
                                            showline=True,mirror=True,
                                            linecolor='black',
                                            linewidth=1,
                                            # backgroundcolor="white",
                                            # gridcolor="black",
                                            showbackground=True
                                            ),
                      xaxis=dict(zeroline=False, showticklabels=False,
                                 ticks="",showgrid=True,title='',
                                 showline=True,mirror=True,
                                 linewidth=1,
                                 linecolor='black',
                                 # backgroundcolor="white",
                                 # gridcolor="black",
                                 showbackground=True
                                 ),
                      zaxis=dict(zeroline=False, showticklabels=False,
                                 ticks="",showgrid=True,title='',
                                 showline=True,mirror=True,
                                 linecolor='black',
                                 linewidth=1,
                                 # backgroundcolor="white",
                                 # gridcolor="black",
                                 showbackground=True
                                 ),
                        ),
                      title_x=0.5,
                      width=width + 50,  # add 50 for colorbar
                      height=int(width * l_0 / l_1),
                      paper_bgcolor='white',
                      coloraxis_colorbar=dict(
                          title="Distance",
                          outlinecolor="black",
                          outlinewidth=2,
                          x=0.9,
                          tickmode='array',
                          tickvals=[0, 0.5, 1, 1.5, 2, 2.5, 3]
                          # thicknessmode='pixels',
                          # thickness=15,
                          # lenmode='pixels',
                          # len=300
                      ),
                       legend=dict(
                           bordercolor='black',
                           borderwidth=1,
                           orientation='h',
                           itemsizing='constant',
                           x=0.10,
                           font=dict(
                               # family="sans-serif",
                               size=10,
                               color="black"
                           ),
                           traceorder='normal'
                       ),
                         # scene_camera_projection_type='orthographic'
                )

    # fig.update_yaxes(automargin=True,showline=True,mirror=True,ticks='inside')
    # fig.update_xaxes(automargin=True,showline=True,mirror=True,ticks='inside')
    # fig.update_zaxes(automargin=True,showline=True,mirror=True)

    epoch = 9999
    plotly.offline.plot(fig, filename="test_som"+str(epoch+1)+".html", image='svg')








