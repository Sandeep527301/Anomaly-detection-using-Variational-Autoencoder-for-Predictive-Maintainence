import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

import tensorflow as tf

# from tensorflow import keras

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.utils import shuffle
from numpy.random import seed

seed(1)
tf.random.set_seed(2)


class SelectThreshold:
    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_train_slim,
        X_val,
        y_val,
        X_val_slim,
        class_to_remove,
        class_names,
        model_name,
        date_time,
    ):

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_slim = X_train_slim
        self.X_val = X_val
        self.y_val = y_val
        self.X_val_slim = X_val_slim
        self.class_to_remove = class_to_remove
        self.class_names = class_names
        self.model_name = model_name
        self.date_time = date_time

    def mse(self, X_val, recon_val):
        """Calculate MSE for images in X_val and recon_val"""
        try:
            return np.mean(np.mean(np.square(X_val - recon_val), axis=1), axis=1)
        except:
            return np.mean(np.square(X_val - recon_val), axis=1)

    def euclidean_distance(self, X_val, recon_val):
        dist = np.linalg.norm(X_val - recon_val, axis=(1, 2))
        return dist

    def create_df_reconstruction(self, y_data, reconstruction_error_val, threshold_val):
        df = pd.DataFrame(data=reconstruction_error_val, columns=["metric"])

        class_names_list = list(zip(self.class_names, range(len(self.class_names))))

        y_names = []
        for i in y_data:
            y_names.append(str(i) + ", " + class_names_list[i][0])

        # append the class values
        df["class"] = y_data
        df["class_names"] = y_names

        # label anomolous (outlier) data as -1, inliers as 1
        # -1 (outlier) is POSITIVE class
        #  1 (inlier) is NEGATIVE class
        new_y_data = []
        for i in y_data:
            if i in self.class_to_remove:
                new_y_data.append(-1)
            else:
                new_y_data.append(1)

        df["true_class"] = new_y_data
        df["prediction"] = np.where(df["metric"] >= threshold_val, -1, 1)

        return df

    def threshold_grid_search(
        self,
        y_data,
        lower_bound,
        upper_bound,
        reconstruction_error_val,
        grid_iterations=10,
    ):
        """Simple grid search for finding the best threshold"""
        roc_scores = {}
        tprs = []  # true positive rates
        fprs = []  # false positive rates
        precisions = []
        recalls = []
        grid_search_count = 0
        for i in np.arange(
            lower_bound,
            upper_bound,
            (np.abs(upper_bound - lower_bound) / grid_iterations),
        ):
            threshold_val = i
            df = self.create_df_reconstruction(y_data, reconstruction_error_val, threshold_val)
            roc_val = roc_auc_score(df["true_class"], df["prediction"])
            roc_scores[i] = roc_val
            grid_search_count += 1
            # calculate precision and recall
            # True Positive
            tp = len(df[(df["true_class"] == -1) & (df["prediction"] == -1)])
            # False Positive -- predict anomaly (-1), when it is actually normal (1)
            fp = len(df[(df["true_class"] == 1) & (df["prediction"] == -1)])
            # True Negative
            tn = len(df[(df["true_class"] == 1) & (df["prediction"] == 1)])
            # False Negative
            fn = len(df[(df["true_class"] == -1) & (df["prediction"] == 1)])
            try:

                # precision/recall
                pre_score = tp / (tp + fp)
                re_score = tp / (tp + fn)
                # tpr/fpr
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)

                precisions.append(pre_score)
                recalls.append(re_score)
                tprs.append(tpr)
                fprs.append(fpr)

            except ZeroDivisionError as err:
                pass

        # return best roc_score and the threshold used to set it
        threshold_val = max(zip(roc_scores.values(), roc_scores.keys()))
        best_threshold = threshold_val[1]
        best_roc_score = threshold_val[0]
        return best_threshold, best_roc_score, precisions, recalls, tprs, fprs

    # function to test the different reconstruction methods (mse, rmse, euclidean)
    # do a grid search looking for the best threshold, and then outputting the results
    def compare_error_method(
        self,
        show_results=True,
        grid_iterations=10,
        model_results=None,
        model_result_cols=[],
        search_iterations=2,
        one_signal_only=False,
        signal_index=None):

        col = [
            "model_name",
            "method",
            "best_threshold",
            "roc_train_score",
            "roc_valid_score",
            "pr_auc_train_score",
            "pr_auc_val_score",
            "date_time",
        ]
        result_table = pd.DataFrame(columns=col)
        for search_iter in range(search_iterations):
            print('search_iter:',search_iter)
            # build the reconstructions on the X_val_slim dataset, and the X_val dataset
            recon_train = self.model.predict(self.X_train, batch_size=64, verbose=1,)
            recon_val = self.model.predict(self.X_val, batch_size=64, verbose=1,) 
            # if we are doing the calculation for one signal only, then:
            if one_signal_only == True:
                mse_recon_train = self.mse(self.X_train[:,:,signal_index], recon_train[:,:,signal_index])  # for complete train dataset
                mse_recon_val = self.mse(self.X_val[:,:,signal_index], recon_val[:,:,signal_index])  # for complete validation dataset

            else:
                mse_recon_train = self.mse(self.X_train, recon_train)  # for complete train dataset
                mse_recon_val = self.mse(self.X_val, recon_val)  # for complete validation dataset

            # calculate pr-auc and roc-auc for train data set
            lower_bound = np.min(mse_recon_train)
            upper_bound = np.max(mse_recon_train)
            (
                best_threshold,
                _,
                precisions,
                recalls,
                tprs,
                fprs,
            ) = self.threshold_grid_search(self.y_train, lower_bound, upper_bound, mse_recon_train, grid_iterations)
            pr_auc_score_train = auc(recalls, precisions)
            roc_auc_score_train = auc(fprs, tprs)

            # calculate pr-auc and roc-auc for train data set
            lower_bound = np.min(mse_recon_val)
            upper_bound = np.max(mse_recon_val)
            _, _, precisions, recalls, tprs, fprs = self.threshold_grid_search(
                self.y_val, lower_bound, upper_bound, mse_recon_val, grid_iterations
            )

            pr_auc_score_val = auc(recalls, precisions)
            roc_auc_score_val = auc(fprs, tprs)
            col = [
                "model_name",
                "method",
                "best_threshold",
                "roc_train_score",
                "roc_valid_score",
                "pr_auc_train_score",
                "pr_auc_val_score",
                "date_time",
            ]
            result_table = pd.concat([result_table,
                pd.DataFrame(
                    [
                        [
                            self.model_name,
                            "mse",
                            best_threshold,
                            roc_auc_score_train,
                            roc_auc_score_val,
                            pr_auc_score_train,
                            pr_auc_score_val,
                            self.date_time,
                        ]
                    ],
                    columns=col,
                )])           
        if model_results == None:
            result_table = result_table.groupby(by=["model_name", "method", "date_time"], as_index=False).mean()

        else:
            result_table = result_table.groupby(by=["model_name", "method", "date_time"], as_index=False).mean()
            result_table = pd.concat([result_table, pd.DataFrame(model_results, columns=model_result_cols)],axis=1,sort=False,)

        if one_signal_only == True:
            result_table['signal_index'] = signal_index
            return result_table
        else:
            return result_table
