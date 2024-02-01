# -*- coding:utf-8 -*-
from loader.load import DataLoader
from model.MLClassifier import MLClassifier
from model.ml_grid import MLGrid

data_select = 'psd_fc_gm_opendata' # 'gm', 'psd_gm', 'psd_fc_gm'
random_seed = 7777

path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\data\\{}'.format(data_select)
loader = DataLoader(path=path, random_seed=random_seed)

# data load
X, y, col = loader.data_load()

grid = MLGrid(random_seed=random_seed)
b_para = grid.knn_grid(x=X, y=y)
print("best parameter:", b_para)

classifier = MLClassifier(random_seed=random_seed, file_name='open')
classifier.knn_best(x=X, y=y, b_para=b_para, columns=col)

# features importance 를 통한 features selection
import pandas as pd
f_im_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\shap_value_save\\open\\'
f_im = pd.read_csv(f_im_path+'shap_f_importance_knn.csv')
f_sel = list(f_im[f_im.columns[0]])
print(f_sel)

# import k fold
from evaluation.k_fold import CVFS
from sklearn.neighbors import KNeighborsClassifier

Kfold = CVFS(random_seed=random_seed, model_name='knn', file_name='knn')
knn_clf = KNeighborsClassifier(
    n_neighbors=19,
    weights='uniform',
    metric='manhattan'
)
df = {'feature len': [], 'auc': [], 'acc': [],'acc_std': [], 'f1': [], 'sensitivity': [], 'specificity': [],
      'PPV': [], 'NPV': []}
for i in range(len(f_sel)):
    print('select number:', i)
    a = f_sel[:i+1]
    x = X[a]
    acc_df = Kfold.k_fold_roc(model=knn_clf, X=x, y=y, fig_save=False)
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
save_path = './auc_save/open/'
df.to_csv(save_path+'f_select_auc_knn.csv')