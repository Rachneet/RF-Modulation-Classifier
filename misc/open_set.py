import h5py as h5
import numpy as np
import scipy.spatial.distance as spd
import libmr
import scipy as sp
import torch
from sklearn import preprocessing
import read_h5 as reader
from scipy import spatial
import statistics
from tqdm import tqdm
import pandas as pd
from dataloader import label_idx
import csv
import matplotlib.pyplot as plt
from scipy import stats


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    mahal=[]
    cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    for i in range(len(data)):
        mahal.append(spd.mahalanobis(x,data[i],inv_covmat))

    return statistics.mean(mahal)


def create_mav(path):
    data = h5.File(path+"feature_set_training_fc8_vsg20.h5", 'r')
    # features,pred_labels,true_labels = data['features'],data['pred_labels'],data['true_labels']
    # data = np.load(path, allow_pickle=True)
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    # print(data.files)
    iq = data['features']
    t_labels = data['true_labels']
    pred_labels = data['pred_labels']

    agg_features = []
    for i in range(len(labels)):
        features = np.zeros(8, )
        count = 0  # reset feature vector and count
        for j in range(iq.shape[0]):
            if pred_labels[j] == t_labels[j]:  # get only the correct features
                if pred_labels[j] == labels[i]:
                    features += iq[j]
                    count += 1
            else:
                continue
        agg_features.append(features / count)

    # print(np.array(agg_features).shape)
    np.savez("/media/backup/Arsenal/rf_dataset_inets/features_fc8_mav.npz",
             features=np.array(agg_features))



def compute_distance(path):
    # compute distance of mav of class from all correct classifications of that class
    data = h5.File(path+"feature_set_training_fc8_vsg20.h5",'r')
    mav_data = np.load(path+"features_fc8_mav.npz")
    topk_mav_data = np.load(path+"features_fc8_topk_mav.npz")
    topk_mav = topk_mav_data['topk']
    mav = mav_data['features']
    features = data['features']
    t_labels = data['true_labels']
    pred_labels = data['pred_labels']
    labels = [0,1,2,3,4,5,6,7]

    for i in range(len(labels)):
        eucos_dist, eu_dist, cos_dist, mahal_dist = [], [], [], []
        for j in range(features.shape[0]):
            if pred_labels[j] == t_labels[j]:  # get correct samples
                if pred_labels[j] == labels[i]:    # get correct class
                    # compute distance between mav and the training example
                    eu_dist += [spd.euclidean(mav[i], features[j])]
                    cos_dist += [spd.cosine(mav[i], features[j])]
                    eucos_dist += [spd.euclidean(mav[i], features[j]) / 200. +
                                   spd.cosine(mav[i], features[j])]
                    # print(pred_labels[j])
                    mahal_dist += [mahalanobis(features[j],topk_mav[i])]
                    # print(mahal_dist)

        # pre-allocate memory
        eu = np.zeros((1, len(eu_dist)), dtype=np.float32)
        cos = np.zeros((1, len(cos_dist)), dtype=np.float32)
        eucos = np.zeros((1, len(eucos_dist)), dtype=np.float32)
        mahal = np.zeros((1, len(mahal_dist)), dtype=np.float32)

        eu[:] = np.array(eu_dist).reshape(1,-1)
        cos[:] = np.array(cos_dist).reshape(1, -1)
        eucos[:] = np.array(eucos_dist).reshape(1, -1)
        mahal[:] = np.array(mahal_dist).reshape(1, -1)

        np.savez(path+"mean_distance_files/class_"+str(i),eu_dist=eu,cos_dist=cos,eucos_dist=eucos,
                 mahal_dist=mahal)

        # channel_distances = {'eucos': eucos, 'cosine': cos, 'euclidean': eu}
        # return channel_distances


def get_top_k_mav(path,k=20):
    data = h5.File(path + "feature_set_training_fc8_vsg20.h5", 'r')
    mav_data = np.load(path + "features_fc8_mav.npz")
    mav = mav_data['features']
    features = data['features']
    t_labels = data['true_labels']
    pred_labels = data['pred_labels']
    labels = [0, 1, 2, 3, 4, 5, 6, 7]
    k_lst = list(range(k + 2))[2:]
    neighbors = []

    for label in labels:
        feature_list = [features[i] for i in range(features.shape[0]) if pred_labels[i] == (t_labels[i] and label)]
        tree = spatial.cKDTree(feature_list)
        dd, idx = tree.query(mav[label], k=k_lst)
        # print(dd,idx)
        topk = [feature_list[i] for i in idx]
        # print(topk)
        closest = np.array(topk).reshape(-1,8)
        neighbors.append(np.vstack((mav[label],closest)))
        # print(neighbors)
        # print(len(neighbors))

    np.savez(path+"features_fc8_topk_mav.npz",topk=np.array(neighbors))


def query_distance_(query_channel, mean_vec, distance_type='eucos'):
    """ Compute the specified distance type between channels of mean vector and query image.
    In torch library, FC8 layer consists of 1 channel. Here, we compute distance
    of distance of this channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used.

    Input:
    --------
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector
    Output:
    --------
    query_distance : Distance between respective channels
    """
    # print(query_channel,mean_vec)
    if distance_type == 'eucos_dist':
        query_distance = spd.euclidean(mean_vec[:], query_channel) / 200. + spd.cosine(mean_vec[:],
                                                                                   query_channel)
    elif distance_type == 'eu_dist':
        query_distance = spd.euclidean(mean_vec[:], query_channel) / 200.
    elif distance_type == 'cos_dist':
        query_distance = spd.cosine(mean_vec[:], query_channel)
    elif distance_type == 'mahal_dist':
        query_distance = mahalanobis(query_channel,mean_vec[:])
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance



def weibull_fitting(path,tailsize=20,distance_type='eucos_dist'):
    # fit the distance distributions
    mav_path = path + "features_fc8_mav.npz"
    topk_mav_path = path + "features_fc8_topk_mav.npz"
    dist_path = path + "mean_distance_files/class_"
    labels = [0,1,2,3,4,5,6,7]
    # data = np.load(path+"mean_distance_files/class_0.npz")
    # print(data['eu_dist'][0])

    weibull_model = {}
    # for each class, read meanfile, distance file, and perform weibull fitting
    for category in labels:
        weibull_model[category] = {}
        distance_scores = np.load(dist_path+str(category)+".npz")[distance_type]
        if distance_type == 'mahal_dist':
            meantrain_vec = np.load(topk_mav_path)['topk'][category]
        else:
            meantrain_vec = np.load(mav_path)['features'][category]

        NCHANNELS = 1
        weibull_model[category]['distances_%s' % distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []
        for channel in range(NCHANNELS):
            mr = libmr.MR()
            tailtofit = sorted(distance_scores[channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category]['weibull_model'] += [mr]

    # print(weibull_model)

    return weibull_model


def query_weibull(class_name, weibull_model, distance_type='eucos_dist'):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]

    Input:
    ------------------------------
    category_name : name of Modulation category as labels
    weibull_model: dictonary of weibull models for
    """

    category_weibull = []
    category_weibull += [weibull_model[class_name]['mean_vec']]
    category_weibull += [weibull_model[class_name]['distances_%s' % distance_type]]
    category_weibull += [weibull_model[class_name]['weibull_model']]

    return category_weibull

NCHANNELS = 1
NCLASSES = 8
ALPHA_RANK = 8
WEIBULL_TAIL_SIZE = 20


def compute_openmax_probability(openmax_fc8,openmax_score_u):
    """ Convert the scores in probability value using openmax

        Input:
        ---------------
        openmax_fc8 : modified FC8 layer from Weibull based computation
        openmax_score_u : degree
        Output:
        ---------------
        modified_scores : probability values modified using OpenMax framework,
        by incorporating degree of uncertainity/openness for a given class

        """

    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            channel_scores += [sp.exp(openmax_fc8[channel, category])]

        total_denominator = sp.sum(sp.exp(openmax_fc8[channel, :])) + sp.exp(sp.sum(openmax_score_u[channel, :]))
        prob_scores += [channel_scores / total_denominator]
        prob_unknowns += [sp.exp(sp.sum(openmax_score_u[channel, :])) / total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    assert len(modified_scores) == 9
    return modified_scores


def recalibrate_scores(weibull_model ,labellist , pred,
                    alpharank=8, distance_type='eucos_dist'):
    """
    Given FC8 features for a signal, list of weibull models for each class,
    re-calibrate scores
    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained from weibull_tailfitting() function
    labellist : class lables as int
    sigarr : features for a particular signal extracted using torch architecture

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using SoftMax (these
    were precomputed from torch architecture. Function returns them for the sake
    of convenience)
    """

    siglayer = pred
    pred = pred.detach().cpu().data.numpy()
    sm = torch.nn.Softmax(dim=1)
    prob = sm(siglayer)
    prob = prob.detach().cpu().data.numpy()
    ranked_list = prob.argsort().ravel()[::-1]   # sort softmax probabilities
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]  # wts for probs
    ranked_alpha = np.zeros(8)

    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = pred[channel, :]
        # print(channel_scores)
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        count = 0
        for categoryid in range(NCLASSES):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(categoryid, weibull_model, distance_type=distance_type)
            # print(category_weibull)
            channel_distance = query_distance_(channel_scores, category_weibull[0],
                                                distance_type=distance_type)

            # obtain w_score for the distance and compute probability of the distance
            # being unknown wrt to mean training vector and channel distances for
            # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            modified_fc8_score = channel_scores[categoryid] * (1 - wscore * ranked_alpha[categoryid])
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score]

        # gather modified scores fc8 scores for each channel for the given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)

    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = compute_openmax_probability(openmax_fc8, openmax_score_u)
    softmax_probab = prob.ravel()
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab)



def main(path):
    model = torch.load("trained_cnn_intf_free_vsg20")
    model.cuda()
    model.eval()
    path_h5 = "/media/backup/Arsenal/rf_dataset_inets/dataset_intf_free_no_cfo_vsg_snr20_1024.h5"
    iq, labels, snrs = reader.read_hdf5(path_h5)
    print("=======Starting======")
    # -----------------------------unk set--------------------------------------------------
    n = 153600
    mods_skip = [8, 9, 18, 19, 23]
    dataset = "/media/backup/Arsenal/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
    h5fr = h5.File(dataset, 'r')
    dset1 = list(h5fr.keys())[0]
    dset2 = list(h5fr.keys())[1]
    dset3 = list(h5fr.keys())[2]
    iq_data = h5fr[dset1][:]
    label_data = h5fr[dset2][:]
    snr_data = h5fr[dset3][:]
    label_idx_ = [label_idx(label) for label in label_data]
    idx_class = [i for i in range(len(label_idx_)) if label_idx_[i] not in mods_skip]
    iq_data = iq_data[idx_class]
    label_data = np.array([np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]) for _ in label_data[idx_class]])
    snr_data = snr_data[idx_class]
    snr_data = snr_data.astype('int8')
    # -----------------------------------------------------------------------------------------------
    df2 = pd.DataFrame()
    df2['iq'] = list(map(lambda x: np.array(x, dtype=np.float32), iq_data))
    df2['labels'] = list(map(lambda x: np.array(x, dtype=np.float32), label_data))
    df2['snr'] = list(map(lambda x: x, snr_data))
    df2 = df2.sample(frac=1, random_state=4)
    print("First df done")

    # analyze data
    df = pd.DataFrame()
    df['iq'] = list(map(lambda x: np.array(x, dtype=np.float32), iq))
    df['labels'] = list(map(lambda x: np.array(x, dtype=np.float32), labels))
    df['snr'] = list(map(lambda x: x, snrs))

    df['labels'] = df.labels.apply(lambda x: np.append(x, np.array([0], dtype=np.float32)))
    print("Second df done")

    print("====start merge====")
    combined_df = pd.concat([df, df2[:n]], ignore_index=True)
    combined_df['normalized_iq'] = combined_df.iq.apply(lambda x: preprocessing.scale(np.array(x), with_mean=False))
    combined_df.drop('iq', axis=1, inplace=True)
    combined_df['labels'] = combined_df.labels.apply(lambda x: np.reshape(x, (-1, 9)))
    print("merged")

    combined_df = combined_df.sample(frac=1, random_state=4)
    print("sampling done")
    print(combined_df.normalized_iq[0].shape)
    print(combined_df.normalized_iq.values[0])
    print(combined_df.labels.values[0].shape)
    print(combined_df.labels.values[0])
    # df = pd.DataFrame()
    # df['iq'] = list(map(lambda x: np.array(x,dtype=np.float32), iq))
    # df['labels'] = list(map(lambda x: np.array(x,dtype=np.float32), labels))
    # df['snr'] = list(map(lambda x: x, snrs))
    # df = df.sample(frac=1, random_state=4)
    # print(file['Y'][200000]                     # np.random.rand(1024,2)            #iq[1][:1024]
    # input = [(i.real,i.imag) for i in input]
    # input = preprocessing.scale(input, with_mean=False)
    # print(input)
    test_bound = int(0.80 * combined_df.labels.shape[0])
    end = int(0.85 * combined_df.labels.shape[0])
    iter = len(combined_df.labels[test_bound:end].values)
    p_label,t_label,snr_label=[],[],[]
    for i in tqdm(range(iter)):
        pred = model(torch.Tensor(combined_df.normalized_iq[test_bound:end].values[i]).unsqueeze(dim=0).cuda())
        # print(pred)
        labellist = [0,1,2,3,4,5,6,7]
        weibull_model = weibull_fitting(path,tailsize=20,distance_type='mahal_dist')
        openmax,softmax = recalibrate_scores(weibull_model,labellist,pred,distance_type='mahal_dist')
        pred_label = np.argmax(openmax)
        true_label = label_idx(np.array(combined_df.labels[test_bound:end].values[i][0]))
        snr = combined_df.snr[test_bound:end].values[i]
        p_label.append(pred_label)
        t_label.append(true_label)
        snr_label.append(snr)

    fieldnames = ['True label', 'Predicted label', 'SNR']
    with open(path + "output_unk_vsg_snr20.csv", 'w', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(t_label, p_label, snr_label):
            writer.writerow(
                {'True label': i, 'Predicted label': j, 'SNR': k})
        # print(openmax,softmax)
        # print("Openmax class: {}".format(np.argmax(openmax)))
        # print("Softmax class: {}".format(np.argmax(softmax)))


if __name__=="__main__":
    path = "/media/backup/Arsenal/rf_dataset_inets/"
    # get_top_k_mav(path)
    # create_mav(path)
    # compute_distance(path)
    # dist = weibull_fitting(path,20,'mahal_dist')[0]['weibull_model']
    # data = weibull_fitting(path,20,'eucos_dist')[0]['mean_vec']
    # x = [2.288636509003411134699e+01 ,1.075142944521545240733e+00 , 6.849604477651003264782e+01,
    #      7.646948210562351633257e+00 ,2.675352631377265311130e+00, 4.320672862326142560363e-01 , 1 ,
    #      5.000000000000000000000e+00 ,1, 20, 1 ,2.175134124755859375000e+02 ,0]
    # plt.plot(data, stats.exponweib.pdf(data, *stats.exponweib.fit(data, 1, 1, scale=2, loc=0)))
    # plt.plot(x)
    # plt.show()
    # print(dist)
    main(path)