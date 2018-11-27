from _0_imports import *


def make_submission(train_pred, test_pred, sub_name):
    sub = pd.read_csv('../input/test.csv')
    label = pd.read_csv('../input/train.csv')['label'].values
    f1 = f1_score(label, np.round(train_pred))
    name = f'../sub/{sub_name}_{f1:.4f}.csv'
    sub['label'] = np.round(test_pred)
    sub.to_csv(name, index=False)


if __name__ == '__main__':
    train_pred = np.load('../stacking/reproduce_all_depth_merged_lgbm_train_0.870753.npy')
    test_pred = np.load('../stacking/reproduce_all_depth_merged_lgbm_test_0.870753.npy')
    make_submission(train_pred, test_pred, 'Merged_LGBM')
