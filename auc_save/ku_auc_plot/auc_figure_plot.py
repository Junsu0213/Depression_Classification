import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# df = pd.read_csv('./features_selection_acc/f_select_acc_auc_(ku_data).csv')
model_name = 'knn'
flist = glob.glob('C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\auc_save\\open\\*{0}.csv'.format(model_name))
# print(flist)

df = pd.read_csv(flist[0])
# print(df)

f_num = df['feature len'].tolist()

auc = df['auc'].tolist()
acc = df['acc'].to_numpy()
acc_std = df['acc_std'].to_numpy()

# print('max auc:', max(auc), '(number:', np.where(np.array(auc) == max(auc))[0][0], ')')
# print('max accuracy:', max(acc), '(number:', np.where(np.array(acc) == max(acc))[0][0], ')')

best_acc_col = np.where(np.array(acc) == max(acc))[0][0]
print('results: {}'.format(model_name))
print(df.iloc[best_acc_col, 1:])


# plt.plot(f_num, auc, label='auc score', linewidth=1)
plt.plot(f_num, acc, 'b-', label='accuracy', linewidth=1)
plt.fill_between(f_num, acc - acc_std, acc + acc_std, color='b', alpha=0.15)

plt.xlabel('Number of features selected')
plt.ylabel('accuracy and auc score')
plt.legend(loc='lower right')
plt.title('{0} feature selection'.format(model_name))
plt.show()