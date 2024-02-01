import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import numpy as np
import pandas as pd


path = "./gm_raw_data/"

# column 만들기
def make_columns(band_name, gm_feature, ch_name):
    node_name = []
    for band in band_name:
        for i in gm_feature:
            for j in ch_name:
                name = band + '-' + i + '-' + j
                node_name.append(name)
    column = node_name
    return column


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
columns = make_columns(band_name=band_name, gm_feature=gm_feature, ch_name=chlist)
print('column len:', len(columns))

# load index
mdd_data = pd.read_csv(path+'MDD_psd_features.csv')
mdd_id = mdd_data.columns[0]
mdd_index = mdd_data[mdd_id]
con_data = pd.read_csv(path+'Control_psd_features.csv')
con_id = con_data.columns[0]
con_index = con_data[con_id]

nodal_measurements = ['1', '7', '13', '44', '15', '21', '27', '33', '35', '36', '39', '42']

for band in band_name:
    xml_file = open(path+'{}.xml'.format(band), 'rt', encoding='UTF8')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    root_test = root[0]

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
                    if gm == nodal_measurements[0] and band == band_name[0]:
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
                    if gm == nodal_measurements[0] and band == band_name[0]:
                        con_gm = con
                    else:
                        con_gm = np.concatenate((con_gm, con), axis=0)
            except KeyError:
                pass

print('mdd shape:', mdd_gm.shape)
print('con shape:', con_gm.shape)

# csv save path
save_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\depression_classificiation\\data\\gm\\'

# mdd dataframe
mdd_gm = mdd_gm.swapaxes(0, 1)
mdd_df = pd.DataFrame(mdd_gm)
mdd_df.columns = columns
mdd_df.insert(0, mdd_id, mdd_index)
mdd_df.to_csv(save_path+'MDD_pli_conn_gm_features.csv')

# control dataframe
con_gm = con_gm.swapaxes(0, 1)
con_df = pd.DataFrame(con_gm)
con_df.columns = columns
con_df.insert(0, con_id, con_index)
con_df.to_csv(save_path+'Control_pli_conn_gm_features.csv')
print(con_df, mdd_df)
