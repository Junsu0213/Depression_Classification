import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import numpy as np
import pandas as pd


path = "./vns_gm/"

file_name = ['delta_theta', 'alpha_beta', 'gamma_result']
band_name = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
gm_feature = ['degree', 'strength', 'triangles', 'eccentricity', 'path length',
              'global efficiency', 'local efficiency', 'clustering coefficient',
              'betweenness centrality', 'closeness centrality', 'within module degree',
              'participation']
chlist = [
    'Fp1', 'F7', 'T7', 'P7', 'O1', 'Fp2',
    'F8', 'T8', 'P8', 'O2', 'F3', 'C3',
    'P3', 'F4', 'C4', 'P4', 'Fz', 'Cz', 'Pz'
]

# load columns
column_df = pd.read_csv('./vns_gm/columns.csv')
columns = column_df.columns

# load index
index_df = pd.read_csv('./vns_gm/index.csv')
id = index_df.columns[0]
index = index_df[id]

nodal_measurements = ['1', '7', '13', '44', '15', '21', '27', '33', '35', '36', '39', '42']

for band in file_name:
    xml_file = open(path+'{}.xml'.format(band), 'rt', encoding='UTF8')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    root_test = root[0]
    print(xml_file)
    if band == file_name[0]:
        for child_ in root_test:
            for gm in nodal_measurements:
                try:
                    if child_.attrib['code'] == gm and child_.attrib['parameters'] == 'NaN' and child_.attrib['group1'] == '1':
                        f = child_.attrib['values1']
                        f = f[1: -1]
                        f = f.replace(';', ' ')
                        f = f.split(" ")
                        f = list(filter(None, f))
                        f = np.array(f, dtype='float')
                        # print(f)
                        mdd = f.reshape((19, -1), order="F")
                        if gm == nodal_measurements[0]:
                            mdd_gm = mdd
                        else:
                            mdd_gm = np.concatenate((mdd_gm, mdd), axis=0)
                    elif child_.attrib['code'] == gm and child_.attrib['parameters'] == 'NaN' and child_.attrib['group1'] == '2':
                        f = child_.attrib['values1']
                        f = f[1: -1]
                        f = f.replace(';', ' ')
                        f = f.split(" ")
                        f = list(filter(None, f))
                        f = np.array(f, dtype='float')
                        con = f.reshape((19, -1), order="F")
                        # print(con.shape)
                        if gm == nodal_measurements[0]:
                            con_gm = con
                        else:
                            con_gm = np.concatenate((con_gm, con), axis=0)
                except KeyError:
                    pass
        delta_theta_gm = np.concatenate((mdd_gm, con_gm), axis=0)
    elif band == file_name[1]:
        for child_ in root_test:
            for gm in nodal_measurements:
                try:
                    if child_.attrib['code'] == gm and child_.attrib['parameters'] == 'NaN' and child_.attrib['group1'] == '1':
                        f = child_.attrib['values1']
                        f = f[1: -1]
                        f = f.replace(';', ' ')
                        f = f.split(" ")
                        f = list(filter(None, f))
                        f = np.array(f, dtype='float')
                        # print(f)
                        mdd = f.reshape((19, -1), order="F")
                        if gm == nodal_measurements[0]:
                            mdd_gm = mdd
                        else:
                            mdd_gm = np.concatenate((mdd_gm, mdd), axis=0)
                    elif child_.attrib['code'] == gm and child_.attrib['parameters'] == 'NaN' and child_.attrib['group1'] == '2':
                        f = child_.attrib['values1']
                        f = f[1: -1]
                        f = f.replace(';', ' ')
                        f = f.split(" ")
                        f = list(filter(None, f))
                        f = np.array(f, dtype='float')
                        con = f.reshape((19, -1), order="F")
                        # print(con.shape)
                        if gm == nodal_measurements[0]:
                            con_gm = con
                        else:
                            con_gm = np.concatenate((con_gm, con), axis=0)
                except KeyError:
                    pass
        alpha_beta_gm = np.concatenate((mdd_gm, con_gm), axis=0)
    elif band == file_name[2]:
        for child_ in root_test:
            for gm in nodal_measurements:
                try:
                    if child_.attrib['code'] == gm and child_.attrib['parameters'] == 'NaN' and child_.attrib['group1'] == '1':
                        f = child_.attrib['values1']
                        f = f[1: -1]
                        f = f.replace(';', ' ')
                        f = f.split(" ")
                        f = list(filter(None, f))
                        f = np.array(f, dtype='float')
                        # print(f)
                        mdd = f.reshape((19, -1), order="F")
                        if gm == nodal_measurements[0]:
                            mdd_gm = mdd
                        else:
                            mdd_gm = np.concatenate((mdd_gm, mdd), axis=0)
                except KeyError:
                    pass
        gamma_gm = mdd_gm

all_data_gm = np.concatenate((delta_theta_gm, alpha_beta_gm, gamma_gm), axis=0)
all_gm = all_data_gm.swapaxes(0, 1)

print(all_gm.shape)
df = pd.DataFrame(all_gm)
df.insert(0, id, index)
df.columns = columns


save_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\depression_classificiation\\data\\vns_csv\\'
df.to_csv(save_path+'vns_pli_conn_gm_features.csv')
#
# print('mdd shape:', mdd_gm.shape)
# print('con shape:', con_gm.shape)
#
# # csv save path
# save_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\depression_classificiation\\data\\gm\\'
#
# # mdd dataframe
# mdd_gm = mdd_gm.swapaxes(0, 1)
# mdd_df = pd.DataFrame(mdd_gm)
# mdd_df.columns = columns
# mdd_df.insert(0, mdd_id, mdd_index)
# mdd_df.to_csv(save_path+'MDD_pli_conn_gm_features.csv')
#
# # control dataframe
# con_gm = con_gm.swapaxes(0, 1)
# con_df = pd.DataFrame(con_gm)
# con_df.columns = columns
# con_df.insert(0, con_id, con_index)
# con_df.to_csv(save_path+'Control_pli_conn_gm_features.csv')
# print(con_df, mdd_df)
