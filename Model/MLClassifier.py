# -*- coding:utf-8 -*-
"""
@author: Jun-su Park
"""
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Model.ml_grid import MLGrid
from Loader.load import DataLoader
from Config.config import DataConfig
from Evaluation.k_fold import CVFS


class MLClassifier(object):
    def __init__(self, config:DataConfig, file_name, fig_save=True):
        self.config = config
        self.random_seed = config.random_seed
        self.MLGrid = MLGrid(random_seed=self.random_seed)
        self.fig_save = fig_save
        self.data_split = DataLoader(config=config, file_name=file_name)
        self.save_path = rf'{config.path}/shap_value_save/{file_name}'
        self.file_name = file_name

    def svm_best(self, x, y, b_para, columns, shap_=True):
        best_para = b_para
        svm_clf = svm.SVC(
            kernel='rbf',
            C=best_para['C'],
            gamma=best_para['gamma'],
            random_state=self.random_seed,
            probability=True
        )
        Kfold = CVFS(config=self.config, model_name='SVM', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=svm_clf, X=x, y=y, fig_save=self.fig_save)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        acc = mean_df['accuracy']
        auc = mean_df['AUC']
        if shap_ == True:
            x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
            svm_clf.fit(X=x_train, y=y_train)
            explainer = shap.KernelExplainer(svm_clf.predict_proba, x_train, link='logit')
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
            mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[0, :, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
            plt.show()
            mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_svm.csv')
        elif shap == False:
            mean_shap_feature_values = None
        return acc, auc, mean_shap_feature_values

    def rf_best(self, x, y, b_para, columns, shap_=True):
        best_para = b_para
        rf_clf = RandomForestClassifier(
            n_estimators=best_para['n_estimators'],
            max_depth=best_para['max_depth'],
            min_samples_leaf=best_para['min_samples_leaf'],
            min_samples_split=best_para['min_samples_split'],
            random_state=self.random_seed,
            n_jobs=-1
        )
        Kfold = CVFS(config=self.config, model_name='Random', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=rf_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        acc = mean_df['accuracy']
        auc = mean_df['AUC']
        if shap_ == True:
            x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
            rf_clf.fit(X=x_train, y=y_train)
            explainer = shap.TreeExplainer(rf_clf)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
            mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[0, :, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
            plt.show()
            mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_rf.csv')
        elif shap_ == False:
            mean_shap_feature_values = None
        return acc, auc, mean_shap_feature_values

    def knn_best(self, x, y, b_para, columns, shap_=True):
        best_para = b_para
        knn_clf = KNeighborsClassifier(
            n_neighbors=best_para['n_neighbors'],
            weights=best_para['weights'],
            metric=best_para['metric'])
        Kfold = CVFS(config=self.config, model_name='KNN', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=knn_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        acc = mean_df['accuracy']
        auc = mean_df['AUC']
        if shap_ == True:
            x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
            knn_clf.fit(X=x_train, y=y_train)
            explainer = shap.KernelExplainer(knn_clf.predict_proba, x_train)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
            mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[0, :, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
            plt.show()
            mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_knn.csv')
        elif shap_ == False:
            mean_shap_feature_values = None
        return acc, auc, mean_shap_feature_values

    def xgb_best(self, x, y, b_para, columns):
        best_para = b_para
        xgb_clf = XGBClassifier(
            n_estimators=best_para['n_estimators'],
            learning_rate=best_para['learning_rate'],
            max_depth=best_para['max_depth'],
            gamma=best_para['gamma'],
            colsample_bytree=best_para['colsample_bytree'])
        Kfold = CVFS(config=self.config, model_name='XGboost', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=xgb_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
        xgb_clf.fit(X=x_train, y=y_train)
        explainer = shap.TreeExplainer(xgb_clf)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
        mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[:, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
        plt.show()
        mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_xgb.csv')
        return mean_shap_feature_values

    def cat_best(self, x, y, b_para, columns):
        best_para = b_para
        cat_clf = CatBoostClassifier(
            iterations=best_para['iterations'],
            depth=best_para['depth'],
            loss_function=best_para['loss_function'],
            l2_leaf_reg=best_para['l2_leaf_reg'],
            leaf_estimation_iterations=best_para['leaf_estimation_iterations'],
            logging_level=best_para['logging_level'],
            random_seed=self.random_seed
        )
        Kfold = CVFS(config=self.config, model_name='atBoost', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=cat_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
        cat_clf.fit(X=x_train, y=y_train)
        explainer = shap.TreeExplainer(cat_clf)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
        mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[:, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
        plt.show()
        mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_cat.csv')
        return mean_shap_feature_values

    def lgbm_best(self, x, y, b_para, columns):
        best_para = b_para
        lgbm_clf = LGBMClassifier(
            n_estimators=200,
            max_depth=best_para['max_depth'],
            min_child_samples=best_para['min_child_samples'],
            subsample=best_para['subsample'])
        Kfold = CVFS(config=self.config, model_name='lgbm', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=lgbm_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
        lgbm_clf.fit(X=x_train, y=y_train)
        explainer = shap.TreeExplainer(lgbm_clf)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
        mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[0, :, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
        plt.show()
        mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_lgbm.csv')
        return mean_shap_feature_values

    def sgd_best(self, x, y, b_para, columns, shap_=True):
        best_para = b_para
        sgd_clf = SGDClassifier(
            loss='modified_huber',
            penalty='elasticnet',
            alpha=best_para['alpha'],
            l1_ratio=best_para['l1_ratio']
        )
        Kfold = CVFS(config=self.config, model_name='SGD', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=sgd_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        acc = mean_df['accuracy']
        auc = mean_df['AUC']
        if shap_ == True:
            x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
            sgd_clf.fit(X=x_train, y=y_train)
            masker = shap.maskers.Independent(data=x_test)
            explainer = shap.LinearExplainer(sgd_clf, masker=masker)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2, plot_type='bar')
            mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[:, :], columns=columns).abs().mean(axis=0).sort_values(ascending=False)
            plt.show()
            mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_sgd.csv')
        elif shap_ == False:
            mean_shap_feature_values = None
        return acc, auc, mean_shap_feature_values

    def lr_best(self, x, y, b_para, columns, shap_=True):
        best_para = b_para
        lr_clf = LogisticRegression(
            penalty=best_para['penalty'],
            C=best_para['C']
        )
        Kfold = CVFS(config=self.config, model_name='logistic regression', file_name=self.file_name)
        acc_df = Kfold.k_fold_roc(model=lr_clf, X=x, y=y, fig_save=self.fig_save)
        print(acc_df)
        mean_df = acc_df.mean()
        print('mean accuracy:', mean_df['accuracy'])
        print('mean AUC:', mean_df['AUC'])
        print('mean f1:', mean_df['macro-f1'])
        print('mean sensitivity:', mean_df['sensitivity'])
        print('mean specificity:', mean_df['specificity'])
        print('mean PPV:', mean_df['PPV'])
        print('mean NPV:', mean_df['NPV'])
        acc = mean_df['accuracy']
        auc = mean_df['AUC']
        if shap_ == True:
            x_train, x_test, y_train, y_test = self.data_split.dataset_split(x=x, y=y)
            lr_clf.fit(X=x_train, y=y_train)
            masker = shap.maskers.Independent(data=x_test)
            explainer = shap.LinearExplainer(lr_clf, masker=masker)
            shap_values = explainer.shap_values(x_test)
            shap.summary_plot(shap_values, x_train, max_display=20, plot_size=(10, 5), alpha=0.2,
                              plot_type='bar')
            mean_shap_feature_values = pd.DataFrame(np.array(shap_values)[:, :], columns=columns).abs().mean(
                axis=0).sort_values(ascending=False)
            plt.show()
            mean_shap_feature_values.to_csv(self.save_path+'shap_f_importance_lr.csv')
        elif shap_ == False:
            mean_shap_feature_values = None
        return acc, auc, mean_shap_feature_values