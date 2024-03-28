# -*- coding:utf-8 -*-
"""
@author: Jun-su Park
"""
import pandas as pd
from Config.config import DataConfig
from Loader.load import DataLoader
from Model.MLClassifier import MLClassifier
from Model.ml_grid import MLGrid
from Evaluation.k_fold import CVFS
from lightgbm import LGBMClassifier


config = DataConfig()
data_select = 'psd_fc_gm_opendata' # 'gm', 'psd_gm', 'psd_fc_gm'
file_name = 'open'

loader = DataLoader(config=config, file_name=data_select)

# data load
X, y, col = loader.data_load()

grid = MLGrid(config=config)
b_para = grid.lgbm_grid(x=X, y=y)
print("best parameter:", b_para)

classifier = MLClassifier(config=config, file_name=file_name)
classifier.lgbm_best(x=X, y=y, b_para=b_para, columns=col)

# features importance 를 통한 features selection
f_im_path = fr'{config.path}/shap_value_save/file_name'
f_im = pd.read_csv(rf'{f_im_path}/shap_f_importance_lgbm.csv')
f_sel = list(f_im[f_im.columns[0]])
print(f_sel)

Kfold = CVFS(config=config, model_name='lgbm', file_name='lgbm')
lgbm_clf = LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    min_child_samples=40,
    subsample=0.8
)
df = {'feature len': [], 'auc': [], 'acc': [],'acc_std': [], 'f1': [], 'sensitivity': [], 'specificity': [],
      'PPV': [], 'NPV': []}
for i in range(len(f_sel)):
    print('select number:', i)
    a = f_sel[:i+1]
    x = X[a]
    acc_df = Kfold.k_fold_roc(model=lgbm_clf, X=x, y=y, fig_save=False)
    mean_df = acc_df.mean()
    std_df = acc_df.std()
    auc = mean_df['AUC']
    acc = mean_df['accuracy']
    acc_std = std_df['accuracy']
    f1 = mean_df['macro-f1']
    sen = mean_df['sensitivity']
    spe = mean_df['specificity']
    ppv = mean_df['PPV']
    npv = mean_df['NPV']
    df['feature len'].append(len(a))
    df['auc'].append(auc)
    df['acc'].append(acc)
    df['acc_std'].append(acc_std)
    df['f1'].append(f1)
    df['sensitivity'].append(sen)
    df['specificity'].append(spe)
    df['PPV'].append(ppv)
    df['NPV'].append(npv)
df = pd.DataFrame(df)
save_path = rf'{config.path}/auc_save/{file_name}'
df.to_csv(rf'{save_path}/f_select_auc_lgbm.csv')
