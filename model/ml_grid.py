import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from ngboost import NGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression, SGDClassifier


class MLGrid(object):
    def __init__(self, random_seed):
        self.random_seed = random_seed

    def svm_grid(self, x, y):
        # 비선형 SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]
        }
        svm_clf = svm.SVC(kernel='rbf')
        clf_grid = GridSearchCV(svm_clf, param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        clf_grid.fit(x, y)
        best_para = clf_grid.best_params_
        return best_para

    def rf_grid(self, x, y):
        params = {
            'n_estimators': range(10, 100, 10),
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 18],
            'min_samples_split': [8, 16, 20]
        }

        rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        clf_grid = GridSearchCV(rf_clf, param_grid=params,scoring="f1_macro", n_jobs=-1, verbose=1)
        clf_grid.fit(x, y)
        best_para = clf_grid.best_params_
        return best_para

    def knn_grid(self, x, y):
        param_grid = {
            'n_neighbors': list(range(1, 20)),
            'weights': ["uniform", "distance"],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        knn_clf = KNeighborsClassifier()
        clf_grid = GridSearchCV(knn_clf, param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        clf_grid.fit(x, y)
        best_para = clf_grid.best_params_
        return best_para

    def lgbm_grid(self, x, y):
        param_grid = {
            'max_depth': [10, 15, 20],
            'min_child_samples': [20, 40, 60],
            'subsample': [0.8, 1]
        }

        lgbm_clf = LGBMClassifier(n_estimators=200)
        clf_grid = GridSearchCV(lgbm_clf, param_grid=param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        clf_grid.fit(x, y)
        best_para = clf_grid.best_params_
        return best_para

    def xgb_grid(self, x, y):
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7, 10, 15],
            'gamma': [0, 1, 2, 3],
            'colsample_bytree': [0.8, 0.9],

        }
        xgb_clf = XGBClassifier()
        clf_grid = GridSearchCV(xgb_clf, param_grid=param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        clf_grid.fit(x, y)
        best_para = clf_grid.best_params_
        return best_para

    def ngb_grid(self, x, y):
        from sklearn.tree import DecisionTreeClassifier
        b1 = DecisionTreeClassifier(criterion='friedman_mse', max_depth=2)
        b2 = DecisionTreeClassifier(criterion='friedman_mse', max_depth=4)
        param_grid = {
            'minibatch_frac': [1.0, 0.5],
            'Base': [b1, b2],
            'n_estimators': [100, 200, 300, 400, 500]
        }
        ngb_clf = NGBClassifier()
        ngb_clf = GridSearchCV(ngb_clf, param_grid=param_grid)
        ngb_clf.fit(x, y)
        best_para = ngb_clf.best_params_
        return best_para

    def cat_grid(self, x, y):
        from sklearn.tree import DecisionTreeClassifier
        param_grid = {
            'iterations': [500],
            'depth': [4, 5, 6],
            'loss_function': ['Logloss', 'CrossEntropy'],
            'l2_leaf_reg': np.logspace(-20, -19, 3),
            'leaf_estimation_iterations': [10],
            'logging_level': ['Silent'],
            'random_seed': [42]
        }

        cat_clf = CatBoostClassifier()
        cat_clf = GridSearchCV(cat_clf, param_grid=param_grid)
        cat_clf.fit(x, y)
        best_para = cat_clf.best_params_
        return best_para

    def lr_grid(self, x, y):
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(0, 4, 10),
        }
        lr_clf = LogisticRegression()
        lr_clf = GridSearchCV(lr_clf, param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        lr_clf.fit(x, y)
        best_para = lr_clf.best_params_
        return best_para

    def sgd_grid(self, x, y):
        param_grid = {
            'alpha': np.logspace(-4, 4, 10),
            'l1_ratio': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2]
        }
        sgd_clf = SGDClassifier()
        sgd_clf = GridSearchCV(sgd_clf, param_grid, scoring="f1_macro", n_jobs=-1, verbose=1)
        sgd_clf.fit(x, y)
        best_para = sgd_clf.best_params_
        return best_para


# if __name__ == '__main__':
#     from Loader.load import DataLoader
#     from ml_model.MLClassifier import MLClassifier
#
#     path = 'C:\\Users\\User\\jupyter\\pipline_depression'
#     random_seed = 777
#     fc_method = 'pli'
#     loader = DataLoader(path=path, random_seed=random_seed)
#
#     # data load
#     X, y = loader.data_load(fc_method=fc_method)
#     # print(X.shape)
#     # print(y.shape)
#
#     grid = MLGrid(random_seed=random_seed)
#     svm_b_para = grid.svm_grid(x=X, y=y)
#     print(svm_b_para)
#
#     classifier = MLClassifier(random_seed=random_seed)
#     classifier.svm_best(x=X, y=y, b_para=svm_b_para)
