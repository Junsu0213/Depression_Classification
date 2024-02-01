# -*- coding:utf-8 -*-
import glob
import pandas as pd
import numpy as np

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, path, random_seed):
        self.path = path
        self.random_seed = random_seed

    def data_load(self, f_select=True):

        # csv file load
        p_flist = glob.glob(self.path+'/MDD*.csv')
        c_flist = glob.glob(self.path + '/Control*.csv')

        # mdd csv file merge
        for flist in p_flist:
            data = pd.read_csv(flist)
            sub_index = data.columns[0]
            index = data[sub_index].values
            data.index = index
            data = data.drop([sub_index], axis=1)
            if flist == p_flist[0]:
                mdd_data = data
            else:
                mdd_data = pd.concat([mdd_data, data], axis=1)

        # control csv files merge
        for flist in c_flist:
            data = pd.read_csv(flist)
            sub_index = data.columns[0]
            index = data[sub_index].values
            data.index = index
            data = data.drop([sub_index], axis=1)
            if flist == c_flist[0]:
                con_data = data
            else:
                con_data = pd.concat([con_data, data], axis=1)

        # mdd, control featurse merge
        all_data = pd.concat([mdd_data, con_data], axis=0)

        if f_select == True:
            # p value 0.05 미만 columns 추출
            t_val, p_val = stats.ttest_ind(mdd_data, con_data, equal_var=False)
            find = np.where(p_val < 0.05)
            columns = all_data.columns
            col_name = columns[find]
            columns = col_name
            print('p value < 0.05', col_name)
            print('len:', len(col_name))

            # all data
            x_ = all_data[list(col_name)]

            # Gaussian distribution으로 scaling
            scaler = StandardScaler()
            x = scaler.fit_transform(x_)

            # data
            x = pd.DataFrame(x, columns=x_.columns, index=list(x_.index.values))

            p_label = np.ones(len(mdd_data.index))
            c_label = np.zeros(len(con_data.index))
            y = np.concatenate((p_label, c_label))
        elif f_select == False:
            # all data
            x_ = all_data

            # Gaussian distribution으로 scaling
            scaler = StandardScaler()
            x = scaler.fit_transform(x_)

            # data
            x = pd.DataFrame(x, columns=x_.columns, index=list(x_.index.values))

            p_label = np.ones(len(mdd_data.index))
            c_label = np.zeros(len(con_data.index))
            y = np.concatenate((p_label, c_label))

            columns = x_.columns

        return x, y, columns

    def dataset_split(self, x, y, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=self.random_seed)

        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data_select = 'psd_fc_gm_ku'  # 'gm', 'psd_gm', 'psd_fc_gm'
    random_seed = 77

    path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\data\\{}'.format(data_select)
    loader = DataLoader(path=path, random_seed=random_seed)

    # data load
    X, y, col = loader.data_load()
    print(X.shape)
    print(y.shape)
    print(col)
    # train, test split
    X_train, X_test, y_train, y_test = loader.dataset_split(x=X, y=y, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)