import pickle as pkl
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import glob
import re
import csv
import pickle


def create_dataset(datapath, list_dir):

    x_data = []
    y_data = []
    df_X, df_y = pd.DataFrame(), pd.DataFrame()
    all_files = []
    sorted_files = []
    for root, dirs, _ in os.walk(datapath):
        for directory in dirs:
            if directory == "pkl":
                files_in_directory = glob.glob(os.path.join(root, directory) + "/*.pkl")
                all_files.extend(files_in_directory)

    for file_path in all_files:
        snr = re.search(r"\d+(\.\d+)?", file_path)
        if int(snr.group(0)) in list_dir:
            sorted_files.append(file_path)

    for file in sorted_files:
        with open(file,'rb') as f:
            out = pkl.load(f)
            x_data.append(out['X'])
            y_data.append(out['y'])
    df_X = pd.concat(x_data).reset_index(drop=True)
    df_y = pd.concat(y_data).reset_index(drop=True)

    return df_X, df_y


def label_id(x):
    mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                   "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
    for i in range(len(mod_schemes)):
        if mod_schemes[i] == x:
            return i


def preprocess_data(df1, df2):  # input: list of dataframes

    features = list(df1.columns)
    scale_features = list(set(features) - set(['SNR']))
    df1.loc[:, scale_features] = preprocessing.scale(df1.loc[:,scale_features])
    df2['label'] = df2['mod_type'] + "_" + df2['mod']
    df2['label'] = df2['label'].apply(lambda x: label_id(x))
    # shuffle datasets
    np.random.seed(4)
    idx = np.random.permutation(df1.index)
    df1 = df1.reindex(idx)
    df2 = df2.reindex(idx)
    return df1, df2[['label','snr_class']]


def classifier(df_x, df_y, save_path, save_result=True):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, shuffle=False)
    D_train = xgb.DMatrix(X_train, label=y_train['label'])
    D_test = xgb.DMatrix(X_test, label=y_test['label'])

    params = {
        'learning_rate': 0.3,   # learning rate, prevents overfitting
        'max_depth': 8,  # depth of decision trees
        'gamma': 0,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'objective': 'multi:softprob',   # loss function
        'num_class': 8}

    steps = 100  # The number of training iterations
    model = xgb.train(params, D_train, steps)

    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(pred) for pred in preds])

    # evaluation metrics
    print("Precision = {}".format(precision_score(y_test['label'], best_preds, average='macro')))
    print("Recall = {}".format(recall_score(y_test['label'], best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test['label'], best_preds)))

    # write results
    if save_result:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save model
        pickle.dump(model, open(save_path + "model.dat", "wb"))
        # save metrics
        fieldnames = ['True_label', 'Predicted_label', 'SNR']
        with open(save_path + "output.csv", 'w', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
            writer.writeheader()
            for i, j, k in zip(y_test['label'], best_preds, y_test['snr_class']):
                writer.writerow(
                    {'True_label': i, 'Predicted_label': j, 'SNR': k})


def grid_cv(df_x, df_y):

    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8)
    clf = xgb.XGBClassifier()
    parameters = {
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],  # shrinks feature values for better boosting
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],   # sum of child weights for further partitioning
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],  # prevents overfitting, split leaf node if min. gamma loss
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7]  # subsample ratio of columns when tree is constructed
    }

    grid = GridSearchCV(clf,
                        parameters, n_jobs=10,
                        scoring="neg_log_loss",
                        cv=3)

    grid.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, grid.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


def main():
    datapath = "/home/rachneet/rf_featurized/vsg_no_cfo/"
    save_path = "/home/rachneet/thesis_results/xg_boost_vsg_all/"
    x_data, y_data = create_dataset(datapath, [0,5,10,15,20])
    x_data, y_data = preprocess_data(x_data, y_data)
    # print(x_data.head())
    # print(y_data.head())
    classifier(x_data, y_data, save_path, save_result=False)











if __name__ == "__main__":
   main()
