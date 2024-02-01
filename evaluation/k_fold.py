import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix


class CVFS(object):
    def __init__(self, random_seed, model_name, file_name):
        self.random_seed = random_seed
        self.model_name = model_name
        self.file_name = file_name
        self.save_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\figure_save\\{}\\'.format(file_name)

    def k_fold_roc(self, model, X, y, n_splits=10, fig_save=True):
        # Run classifier with cross-validation and plot ROC curves
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        X = np.array(X)
        y = np.array(y)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(9.5, 9.5))
        result_df = {'accuracy': [], 'AUC': [],
                     'macro-f1': [], 'sensitivity': [], 'specificity': [], 'PPV': [], 'NPV': []}
        i = 1
        for train, test in cv.split(X, y):
            y_pred = model.fit(X[train], y[train]).predict(X[test])
            y_proba = model.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], y_proba[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1

            acc, mf1, = accuracy_score(y_pred, y[test]), f1_score(y_true=y[test], y_pred=y_pred, average='macro')
            auc_score = roc_auc_score(y_true=y[test], y_score=y_proba[:, 1], average='macro')
            tn, fp, fn, tp = confusion_matrix(y[test], y_pred, labels=[0, 1]).ravel()
            sen = tp / (tp + fn)
            spe = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (fn + tn)

            result_df['AUC'].append(auc_score)
            result_df['accuracy'].append(acc)
            result_df['macro-f1'].append(mf1)
            result_df['sensitivity'].append(sen)
            result_df['specificity'].append(spe)
            result_df['PPV'].append(ppv)
            result_df['NPV'].append(npv)

        result_df = pd.DataFrame(result_df)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('Cross-Validation ROC of {0} ({1})'.format(self.model_name, self.file_name), fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        if fig_save == True:
            plt.savefig(self.save_path+'k fold {0} {1}'.format(self.model_name, self.file_name))
            plt.show()
        elif fig_save is not True:
            pass
        return result_df

    def fs_plot(self):
        f_select_path = 'C:\\Users\\User\\Desktop\\PyCharm Projects\\Depression_classification_final\\shap_value_save\\'
        df = pd.read_csv(f_select_path+'f_select_acc_auc_{0}_{1}.csv'.format(self.model_name, self.file_name))
        print(df)

        f_num = df['feature len'].tolist()

        auc = df['auc'].tolist()
        acc = df['accuracy'].tolist()

        print('max auc:', max(auc), '(number:', np.where(np.array(auc) == max(auc))[0][0], ')')
        print('max accuracy:', max(acc), '(number:', np.where(np.array(acc) == max(acc))[0][0], ')')

        plt.plot(f_num, auc, label='auc score', linewidth=1)
        plt.plot(f_num, acc, label='accuracy', linewidth=1)

        plt.xlabel('Number of features selected')
        plt.ylabel('accuracy and auc score')
        plt.legend(loc='lower right')
        plt.title('feature selection')
        plt.savefig(self.save_path + 'Shap features selection {0} {1}'.format(self.model_name, self.file_name))
        plt.show()