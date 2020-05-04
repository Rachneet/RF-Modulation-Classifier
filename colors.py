from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

color_array = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
               (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
               (1.0, 0.4980392156862745, 0.054901960784313725),
               (1.0, 0.7333333333333333, 0.47058823529411764),
               (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
               (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
               (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
               (1.0, 0.596078431372549, 0.5882352941176471),
               (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
               (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
               (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
               (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
               (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
               (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
               (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
               (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
               (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
               (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
               (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
               (0.6196078431372549, 0.8549019607843137, 0.8980392156862745)]

color_array_solid = color_array[::2]
color_array_shade = color_array[1::2]

cm_confusion = LinearSegmentedColormap.from_list(name='confusion',
    colors=[color_array[0], color_array[2]], N=100)
cm_colors = LinearSegmentedColormap.from_list(name='colors',
    colors=color_array, N=len(color_array))
cm_confusion.set_bad((1, 0, 0, 0))

color_bw_signal = tuple(np.array([210, 242, 201]) / 255)
color_bw_noise = tuple(np.array([255, 201, 201]) / 255)

#cmap_list = [(1, 1, 1), color_array[2], color_array[6], color_array[8], color_array[0], color_array[4]]  # R -> G -> B
cmap_list = [(1, 1, 1), color_array[6], color_array[4]]  # R -> G -> B
cmap_conf = LinearSegmentedColormap.from_list("cmap_conf", cmap_list, N=100)


def show_colors():
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111)
    matrix = np.reshape(np.arange(len(color_array)), (-1, 10))
    ax.imshow(np.array(matrix), interpolation='nearest', cmap=cm_colors)
    ax.set_axis_off()
    plt.show()
