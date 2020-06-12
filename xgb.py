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
import shap
import matplotlib.pyplot as plt
import warnings


class XgbModule(object):
    def __init__(self, data_path, save_path, save_results=True):

        shap.initjs()
        self.data_path = data_path
        self.save_path = save_path
        self.mod_schemes = ["SC_BPSK", "SC_QPSK", "SC_16QAM", "SC_64QAM",
                       "OFDM_BPSK", "OFDM_QPSK", "OFDM_16QAM", "OFDM_64QAM"]
        self.snr = [0, 5, 10, 15, 20]
        self.save_results = save_results


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

        for file_path in all_files:
            snr = re.search(r"\d+(\.\d+)?", file_path)
            if int(snr.group(0)) in self.snr:
                sorted_files.append(file_path)

        for file in sorted_files:
            with open(file,'rb') as f:
                out = pkl.load(f)
                x_data.append(out['X'])
                y_data.append(out['y'])
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
        df2['label'] = df2['mod_type'] + "_" + df2['mod']
        df2['label'] = df2['label'].apply(lambda x: self.label_id(x))
        # shuffle datasets
        np.random.seed(4)
        idx = np.random.permutation(df1.index)
        df1 = df1.reindex(idx)
        # print(df1.columns)
        #
        # df1.drop(['$\\gamma_{4,max}$'], axis=1, inplace=True)
        df2 = df2.reindex(idx)
        return df1, df2[['label','snr_class']]


    def classifier(self, df_x, df_y):
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
        if self.save_results:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # save model
            pickle.dump(model, open(self.save_path + "model.dat", "wb"))
            # save metrics
            fieldnames = ['True_label', 'Predicted_label', 'SNR']
            with open(self.save_path + "output.csv", 'w', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                writer.writeheader()
                for i, j, k in zip(y_test['label'], best_preds, y_test['snr_class']):
                    writer.writerow(
                        {'True_label': i, 'Predicted_label': j, 'SNR': k})


    def grid_cv(self, df_x, df_y):

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


    def class_labels(self, y_test):
        return [f'{self.mod_schemes[i]} ({y_test[i].round(2):.2f})'
        for i in range(len(self.mod_schemes))]


    def get_shape_values(self, model, df_x, df_y):
        X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.8, shuffle=False)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        # print(X_test.iloc[0, :])
        # compute the SHAP values for every prediction in the test dataset
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        expected_values = explainer.expected_value

        return expected_values, shap_values, X_test, y_test


    def shap_feature_importance(self, model, df_x,df_y):
        _, shap_values, X_test, _ = self.get_shape_values(model, df_x, df_y)
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar",color=plt.get_cmap("tab10"),
                                class_names=self.mod_schemes)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [7, 6, 3, 2, 4, 5, 1, 0]
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(self.save_path + "shap_summary.svg")


    def shap_decision_plot(self, model, df_x, df_y):

        expected_values, shap_values, X_test, y_test = self.get_shape_values(model, df_x, df_y)

        row_index = 45
        y_pred = model.predict(xgb.DMatrix(X_test))
        # print(y_test['label'][:50])
        # print([np.argmax(pred) for pred in y_pred[:50]])
        # print(list(df_x.columns))
        # print(np.array(shap_values)[2][0])
        shap.multioutput_decision_plot(expected_values, shap_values,
                                       row_index=row_index,
                                       highlight=[y_test['label'][row_index]],
                                       feature_names=list(df_x.columns),
                                       legend_labels=self.class_labels(y_pred[row_index]),
                                       legend_location='lower right', show=False)
        # plt.show()
        plt.savefig(self.save_path + "shap_decision_wrong_pred.svg")


    def identify_outliers(self, model, df_x, df_y):
        expected_values, _, X_test, y_test = self.get_shape_values(model, df_x, df_y)
        y_pred = model.predict(xgb.DMatrix(X_test))
        T = X_test[(y_pred>0.03)&(y_pred<0.1)]
        T = T.reset_index(drop=True)
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sh = explainer.shap_values(T)

        r = shap.multioutput_decision_plot(expected_values, sh, T, feature_order='hclust',
                                           return_objects=True)
        plt.show()


    def main(self):
        x_data, y_data = self.create_dataset()
        x_data, y_data = self.preprocess_data(x_data, y_data)
        # print(x_data.head())
        # print(y_data.head())
        # self.classifier(x_data, y_data)
        model = pickle.load(open(self.save_path+"model.dat",'rb'))
        self.identify_outliers(model, x_data, y_data)


if __name__ == "__main__":
    datapath = "/home/rachneet/rf_featurized/vsg_no_cfo/"
    save_path = "/home/rachneet/thesis_results/xg_boost_vsg_all/"
    xgb_obj = XgbModule(datapath, save_path, save_results=False)
    xgb_obj.main()