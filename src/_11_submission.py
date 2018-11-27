from tqdm import tqdm

from _0_imports import *
from _8_ensemble import blending


def make_submission(train_pred, test_pred, sub_name):
    sub = pd.read_csv('../input/test.csv')
    label = pd.read_csv('../input/train.csv')['label'].values
    f1 = f1_score(label, np.round(train_pred))
    name = f'../sub/{sub_name}_{f1:.4f}.csv'
    sub['label'] = np.round(test_pred)
    sub.to_csv(name, index=False)


if __name__ == '__main__':
    # CV: 0.8572 LB: 0.85251
    # train_pred, test_pred = blending('../oof/InferSent', f1_or_loss='f1')
    # make_submission(train_pred, test_pred, '10_InferSent')

    # train_pred, test_pred = blending('../oof/SiameseTextCNN', f1_or_loss='f1')
    # make_submission(train_pred, test_pred, '10_SiameseTextCNN')

    # train_pred, test_pred = blending('../oof/DecomposableAttention', f1_or_loss='f1')
    # make_submission(train_pred, test_pred, '10_DecomposableAttention')

    # train_pred, test_pred = blending('../oof/InferSent_Feat', f1_or_loss='both')
    # make_submission(train_pred, test_pred, '10_InferSent_Feat')

    # train_pred, test_pred = blending('../oof/InferSent_tuneEmbed', f1_or_loss='both')
    # make_submission(train_pred, test_pred, '10_InferSent_tuneEmbed')

    # train_pred, test_pred = blending('../oof/InferSent_SGD', f1_or_loss='f1')
    # make_submission(train_pred, test_pred, '20_InferSent_SGD')

    # train_pred, test_pred = np.load('../stacking/reproduce_all_depth_merged_lgbm_train_0.870753.npy'), np.load('../stacking/reproduce_all_depth_merged_lgbm_test_0.870753.npy')
    # make_submission(train_pred, test_pred, 'Merged_LGBM')

    train_pred, test_pred = np.load('../stacking/reproduce_all_depth_merged_lgbm_1125_train_0.872423.npy'), np.load(
        '../stacking/reproduce_all_depth_merged_lgbm_1125_test_0.872423.npy')
    make_submission(train_pred, test_pred, 'Merged_LGBM_1125')

# train_pred, test_pred = blending('../oof/InferSent_LSTM_with_PL')
# make_submission(train_pred, test_pred, '10_InferSent_LSTM_with_PL')
# train_pred, test_pred = blending('../oof/InferSent_LSTM')
# make_submission(train_pred, test_pred, '10_InferSent_LSTM_onlyF1')
# train_pred, test_pred = blending('../oof/InferSent_LSTM_Char')
# make_submission(train_pred, test_pred, '5_InferSent_LSTM_Char')

# train_pred, test_pred = blending('../oof/InferSent_Concat')
# make_submission(train_pred, test_pred, '5_InferSent_LSTM_Concat')

# train_pred, test_pred = blending('../oof/1DCNN')
# make_submission(train_pred, test_pred, '6_1DCNN')


# train_pred, test_pred = blending('../oof/DecomposableAttention')
# make_submission(train_pred, test_pred, '5_DecomposableAttention')
