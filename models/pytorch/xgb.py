import pickle as pkl
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import glob
import re
import csv
import pickle
# import shap
import matplotlib.pyplot as plt
import warnings
from xgboost import plot_tree
from inference.inference import compute_results
import data_processing.dataloader as dl
from data_processing.merge_filtered import *


class XgbModule(object):
    def __init__(self, data_path, save_path, train_size, save_results=True):

        # shap.initjs()
        self.data_path = data_path
        self.save_path = save_path
        self.train_size = train_size
        # self.mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
        #                "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
        self.mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM"]
        self.mods = ["SC BPSK", "SC QPSK", "SC 16-QAM", "SC 64-QAM",
                "OFDM BPSK", "OFDM QPSK", "OFDM 16-QAM", "OFDM 64-QAM"]
        self.snr = [0, 5, 10, 15, 20]
        self.save_results = save_results


    def create_deepsig_set(self):
        # if self.data_path[-3:] == "csv":
        df = pd.read_csv(self.data_path)
        df_X = df.drop(['label'], axis=1)
        df_y = df[['SNR','label']]

        features = list(df_X.columns)
        scale_features = list(set(features) - set(['SNR']))
        df_X.loc[:, scale_features] = preprocessing.scale(df_X.loc[:, scale_features])
        # shuffle datasets
        np.random.seed(4)
        idx = np.random.permutation(df_X.index)
        df_X = df_X.reindex(idx)
        df_y = df_y.reindex(idx)
        print('-----------------dataset created---------------')
        return df_X, df_y


    def create_dataset(self):
        x_data = []
        y_data = []
        all_files = []
        sorted_files = []
        for root, dirs, _ in os.walk(self.data_path):
            for directory in dirs:
                if directory == "pkl":
                    files_in_directory = glob.glob(os.path.join(root, directory) + "/*.pkl")
                    all_files.extend(files_in_directory)
        # print(all_files)

        for file_path in all_files:
            snr = re.search(r"\d+(\.\d+)?", file_path)
            # print(snr.group(1))
            if int(snr.group(0)) in self.snr:    # remember change
                sorted_files.append(file_path)
                print(sorted_files)

        for file in sorted_files:
            with open(file,'rb') as f:
                out = pkl.load(f)
                x_data.append(out['X'])
                y_data.append(out['y'])
        print(x_data)
        df_X = pd.concat(x_data).reset_index(drop=True)
        df_y = pd.concat(y_data).reset_index(drop=True)

        return df_X, df_y


    def label_id(self, x):

        for i in range(len(self.mod_schemes)):
            if self.mod_schemes[i] == x:
                return i


    def preprocess_data(self, df1, df2):   # input: list of dataframes

        features = list(df1.columns)
        scale_features = list(set(features) - set(['SNR']))
        df1.loc[:, scale_features] = preprocessing.scale(df1.loc[:,scale_features])
        # print(list(df1.columns))
        # df1 = df1.rename(columns={'$\\mu^A_{42}$':'$\\mu^a_{42}$'})
        # print(list(df1.columns))
        df2['label'] = df2['mod_type'] + "_" + df2['mod']
        df2['label'] = df2['label'].apply(lambda x: self.label_id(x))
        # df2['label'] = df2.loc[df2['label'].isin([0,1,2,3])]
        # shuffle datasets
        np.random.seed(4)
        idx = np.random.permutation(df1.index)
        df1 = df1.reindex(idx)
        # print(df1.columns)
        #
        # df1.drop(['$\\gamma_{4,max}$'], axis=1, inplace=True)
        df2 = df2.reindex(idx)
        return df1, df2[['label','snr_class']]

    def classifier(self, df_x, df_y, do_train=True, do_predict=True):
        test_size = 0.2
        X_tr, X_test, y_tr, y_test = train_test_split(df_x, df_y, test_size=test_size, shuffle=False)
        # X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.0625, shuffle=False)
        train_size = self.train_size/(1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, train_size=train_size, shuffle=False)
        params = {
            'learning_rate': 0.2,  # learning rate, prevents overfitting
            'max_depth': 15,  # depth of decision trees
            'gamma': 0.4,
            'colsample_bytree': 0.3,
            'min_child_weight': 3,
            'objective': 'multi:softprob',  # loss function
            'num_class': 8}

        best_preds = np.array([])
        if do_train and do_predict:
            D_train = xg.DMatrix(X_train, label=y_train['label'])
            D_val = xg.DMatrix(X_val, label=y_val['label'])
            D_test = xg.DMatrix(X_test, label=y_test['label'])
            steps = 200  # The number of training iterations
            evals = [(D_val, "validation")]
            model = xg.train(params, D_train, num_boost_round=steps, evals=evals,  # early_stopping_rounds=10,
                             verbose_eval=True)
            preds = model.predict(D_test)
            best_preds = np.asarray([np.argmax(pred) for pred in preds])
        elif do_train:
            D_train = xg.DMatrix(X_train, label=y_train['label'])
            D_val = xg.DMatrix(X_val, label=y_val['label'])
            steps = 200  # The number of training iterations
            evals = [(D_val, "validation")]
            model = xg.train(params, D_train, num_boost_round=steps, evals=evals,  # early_stopping_rounds=10,
                             verbose_eval=True)
        elif do_predict:
            D_test = xg.DMatrix(X_test, label=y_test['label'])
            model = pickle.load(open(self.save_path + "model.dat", "rb"))
            preds = model.predict(D_test)
            best_preds = np.asarray([np.argmax(pred) for pred in preds])
        else:
            print("Select whether to train or test or both")

        # evaluation metrics
        print("Precision = {}".format(precision_score(y_test['label'], best_preds, average='macro')))
        print("Recall = {}".format(recall_score(y_test['label'], best_preds, average='macro')))
        print("Accuracy = {}".format(accuracy_score(y_test['label'], best_preds)))
        print("Confusion matrix = {}".format(confusion_matrix(y_test['label'], best_preds)))

        # write results
        if self.save_results:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # save model
            if do_train:
                pickle.dump(model, open(self.save_path + "model.dat", "wb"))
            # save metrics
            fieldnames = ['True_label', 'Predicted_label', 'SNR']
            with open(self.save_path + "output.csv", 'w', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                for i, j, k in zip(y_test['label'], best_preds, y_test['SNR']):  # rem change
                    writer.writerow(
                        {'True_label': i, 'Predicted_label': j, 'SNR': k})


    def read_feature_file(self, path):
        file = h5.File(path, 'r')
        features, labels = file['features'], file['true_labels']
        return features, labels

    def train_xgb_cnn(self):
        path = "/home/rachneet/rf_dataset_inets/"
        # X_train, y_train = self.read_feature_file(path + "feature_set_training_fc8_vsg_all.h5")
        X_val, y_val = self.read_feature_file(path + "feature_set_training_fc8_vsg_all_val.h5")
        print(y_val[0])
        # X_test, y_test = self.read_feature_file(path + "feature_set_training_fc8_vsg_all_test.h5")

        # D_train = xg.DMatrix(X_train, label=y_train)
        # D_val = xg.DMatrix(X_val, label=y_val)
        # D_test = xg.DMatrix(X_test, label=y_test)
        #
        # params = {
        #     'learning_rate': 0.2,  # learning rate, prevents overfitting
        #     'max_depth': 18,  # depth of decision trees
        #     'gamma': 0.4,
        #     'colsample_bytree': 0.3,
        #     'min_child_weight': 3,
        #     'objective': 'multi:softprob',  # loss function
        #     'num_class': 8}
        # # 'gpu_id': 0,
        # # 'tree_method': 'gpu_hist'}
        #
        # steps = 200  # The number of training iterations
        # evals = [(D_val, "validation")]
        # model = xg.train(params, D_train, num_boost_round=steps, evals=evals, early_stopping_rounds=10,
        #                  verbose_eval=True)
        #
        # preds = model.predict(D_test)
        # best_preds = np.asarray([np.argmax(pred) for pred in preds])
        #
        # # evaluation metrics
        # print("Precision = {}".format(precision_score(y_test['label'], best_preds, average='macro')))
        # print("Recall = {}".format(recall_score(y_test['label'], best_preds, average='macro')))
        # print("Accuracy = {}".format(accuracy_score(y_test['label'], best_preds)))


    def cross_validate_model(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, shuffle=False)
        D_train = xg.DMatrix(X_train, label=y_train['label'])
        D_test = xg.DMatrix(X_test, label=y_test['label'])

        params = {
            'learning_rate': 0.2,  # learning rate, prevents overfitting
            'max_depth': 18,  # depth of decision trees
            'gamma': 0.4,
            'colsample_bytree': 0.3,
            'min_child_weight': 3,
            'objective': 'multi:softprob',  # loss function
            'num_class': 8}

        # steps = 100  # The number of training iterations
        # model = xgb.train(params, D_train, steps, evals=[(D_test, "Test")], early_stopping_rounds=10)
        # preds = model.predict(D_test)
        # best_preds = np.asarray([np.argmax(pred) for pred in preds])
        #
        # # evaluation metrics
        # print("Precision = {}".format(precision_score(y_test['label'], best_preds, average='macro')))
        # print("Recall = {}".format(recall_score(y_test['label'], best_preds, average='macro')))
        # print("Accuracy = {}".format(accuracy_score(y_test['label'], best_preds)))
        # Run CV
        cv_results = xg.cv(
            params,
            D_train,
            num_boost_round=100,
            seed=4,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_merror = cv_results['test-merror-mean'].min()
        std_merror = cv_results['test-merror-std'].min()
        print(mean_merror, std_merror)


    def grid_cv(self, df_x, df_y):

        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.2)
        y_train = y_train['label']
        y_test = y_test['label']

        clf = xg.XGBClassifier()
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
        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)


    def class_labels(self, y_test):
        return [f'{self.mods[i]} ({y_test[i].round(2):.2f})'
        for i in range(len(self.mods))]


    # def get_shape_values(self, model, df_x, df_y):
    #     X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, shuffle=False)
    #     X_test = X_test.reset_index(drop=True)
    #     y_test = y_test.reset_index(drop=True)
    #     # print(X_test.iloc[0, :])
    #     # compute the SHAP values for every prediction in the test dataset
    #     explainer = shap.TreeExplainer(model)
    #     shap_values = explainer.shap_values(X_test)
    #     expected_values = explainer.expected_value
    #
    #     return expected_values, shap_values, X_test, y_test


    # def shap_feature_importance(self, model, df_x,df_y):
    #     _, shap_values, X_test, _ = self.get_shape_values(model, df_x, df_y)
    #     # print(shap_values)
    #     shap.summary_plot(shap_values, X_test, show=False, plot_type="bar",color=plt.get_cmap("tab10"),
    #                             class_names=self.mods)
    #
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     order = [7, 6, 4, 2, 1, 5, 3, 0]
    #     plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    #     # plt.ylabel("Features", fontsize=13)
    #     plt.xlabel("")
    #     # plt.xticks(np.arange(0, 12, step=2))
    #     locs, labels = plt.xticks()
    #
    #     # print(locs, labels)
    #     plt.xticks(locs, [0, 2, 4, 6, 8, 10, 12], fontsize=14)
    #     plt.yticks(fontsize=14)
    #     plt.gca().spines['right'].set_visible(True)
    #     plt.gca().spines['top'].set_visible(True)
    #     plt.gca().spines['left'].set_visible(True)
    #     # plt.show()
    #     plt.savefig(self.save_path + "shap_summary.svg")


    # def shap_decision_plot(self, model, df_x, df_y):
    #
    #     print(df_x.columns)
    #     expected_values, shap_values, X_test, y_test = self.get_shape_values(model, df_x, df_y)
    #     # 2 qpsk 8  45
    #     row_index = 2
    #     y_pred = model.predict(xg.DMatrix(X_test))
    #     # print(y_test['label'][:50])
    #     # print([np.argmax(pred) for pred in y_pred[:50]])
    #     # print(list(df_x.columns))
    #     # print(np.array(shap_values)[2][0])
    #
    #     shap.multioutput_decision_plot(expected_values, shap_values,
    #                                    row_index=row_index,
    #                                    highlight=[y_test['label'][row_index]],
    #                                    feature_names=list(df_x.columns),
    #                                    legend_labels=self.class_labels(y_pred[row_index]),
    #                                    legend_location='lower right', show=False,
    #                                    feature_display_range=slice(None, -11, -1))
    #     locs_y, labels_y = plt.yticks()
    #     print(locs_y, labels_y)
    #     plt.xlabel("")
    #     plt.gca().spines['right'].set_visible(True)
    #     plt.gca().spines['top'].set_visible(True)
    #     plt.gca().spines['left'].set_visible(True)
    #     plt.xticks(fontsize=14)
        # plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] ,['$ZC$', '$\\sigma_{aa}$', '$\\gamma_{2,max}$', '$\\sigma_{a}$',
        #                                                                '$\\Psi_{max}$', '$\\mu^a_{42}$', '$|\\widetilde{C}_{40}|$',
        # '$\\widetilde{C}_{63}$', '$\\widetilde{C}_{42}$', '$\\mu^R_{42}$'], fontsize=14)
        # plt.show()
        # plt.savefig(self.save_path + "shap_decision_qpsk.svg")


    # def identify_outliers(self, model, df_x, df_y):
    #     expected_values, _, X_test, y_test = self.get_shape_values(model, df_x, df_y)
    #     y_pred = model.predict(xgb.DMatrix(X_test))
    #     T = X_test[(y_pred>0.03)&(y_pred<0.1)]
    #     T = T.reset_index(drop=True)
    #     explainer = shap.TreeExplainer(model)
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         sh = explainer.shap_values(T)
    #     r = shap.multioutput_decision_plot(expected_values, sh, T, feature_order='hclust',
    #                                        return_objects=True)
    #     plt.show()


    # def dependence_plot(self, model, df_x, df_y):
    #     print(list(df_x.columns))
    #     expected_values, shap_values, X_test, y_test = self.get_shape_values(model, df_x, df_y)
    #     # np.seterr(divide='ignore', invalid='ignore')
    #     with warnings.catch_warnings():
    #         warnings.simplefilter('ignore')
    #         # s = np.array(shap_values)
    #         # s[s==0] = 0.1
    #         # s = list(s)
    #         # shap.dependence_plot('$\\mu^R_{42}$', shap_values[5], X_test,
    #         #                      show=False)# ,interaction_index='$\\sigma_{ap,CNL}$')
    #         shap.dependence_plot('$\\widetilde{C}_{42}$', shap_values[3], X_test,
    #                              show=False)#, interaction_index='$|\\widetilde{C}_{40}|$')
    #
    #     figure = plt.gcf()
    #     figure.set_size_inches(4.6, 3)
    #     plt.tight_layout()
    #     # cb = plt.colorbar(x)
    #     # cb.set_ylabel("")
    #
    #     plt.gca().spines['right'].set_visible(True)
    #     plt.gca().spines['top'].set_visible(True)
    #     # plt.xticks(fontsize=14)
    #     # plt.yticks(fontsize=14)
    #     # plt.xlabel("")
    #     # plt.ylabel("")
    #
    #     # plt.show()
    #     plt.savefig(self.save_path + "shap_dep_sc_64qam.svg")

    # main file
    def main(self):
        # self.train_xgb_cnn()
        x_data, y_data = self.create_deepsig_set()  # note: pre-processing done in-situ
        # x_data, y_data = self.preprocess_data(x_data, y_data)
        # self.grid_cv(x_data, y_data)
        self.classifier(x_data, y_data, do_train=True, do_predict=True)
        # model = pickle.load(open(self.save_path+"model.dat",'rb'))
        # self.shap_feature_importance(model, x_data, y_data)
        # self.shap_decision_plot(model, x_data, y_data)
        # # self.cross_validate_model(x_data, y_data)
        # self.dependence_plot(model, x_data, y_data)
        # plot_tree(model)
        # plt.show()


def split_csv():
    # classes = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK',
    #            '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC',
    #            'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
    classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK',
               '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
               'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
    norm_classes = ['OOK', '4ASK', 'BPSK', 'QPSK', '8PSK', '16QAM', 'AM-SSB-SC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
    digital_mods = ['BPSK', 'QPSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
    vier_mods = ['BPSK', 'QPSK', '16QAM', '64QAM']
    label_list = []

    for i in vier_mods:
        for j in range(len(classes)):
            if i == classes[j]:
                label_list.append(j)
    # print(label_list)
    path = "/home/rachneet/rf_featurized/"
    df = pd.read_csv(path + "deepsig_featurized_set.csv")
    df = df[df['label'].isin(label_list)]
    df['label'] = df['label'].apply(lambda x:mod_fix(x, label_list))
    # print(df.head())
    df.to_csv(path+"deepsig_featurized_vier_mod.csv", index=False)
    # t_label = sorted(list(pd.unique(df['True_label'].values)))
    # print(t_label)
    # print(df.head())


def mod_fix(label, label_list):
    for l in range(len(label_list)):
        if label == label_list[l]:
            return l


def explore_results():
    path = "/home/rachneet/thesis_results/deepsig_results_11mod/"
    df = pd.read_csv(path+"output.csv")
    snr = sorted(list(pd.unique(df['SNR'].values)))
    t_label = sorted(list(pd.unique(df['True_label'].values)))
    count, result = compute_results(path+"output.csv", snr)
    for k, v in result.items():
        print(k,":",v['accuracy'])


if __name__ == "__main__":
    datapath = "/home/rachneet/featurized_data/dataset_vsg_no_intf_featurized.csv"
    train_sizes = [0.60, 0.45, 0.30, 0.15]
    for i in range(len(train_sizes)):
        str_train_size = "{:.2f}".format(train_sizes[i])
        save_path = f"/home/rachneet/DATA/results/xgb_vsg_{str_train_size.split('.')[-1]}/"
        xgb_obj = XgbModule(datapath, save_path, train_size=train_sizes[i], save_results=True)
        xgb_obj.main()