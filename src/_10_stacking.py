from lightgbm import LGBMClassifier

from _0_imports import *


def get_layer1_input():
    # decatt_train = np.load('../blending/DecomposableAttention_1125_f1_train_0.8527.npy')
    # decatt_test = np.load('../blending/DecomposableAttention_1125_f1_test_0.8527.npy')

    # decatt_feat_train = np.load('../blending/DecomposableAttention_Feat_1125_f1_train_0.8622.npy')
    # decatt_feat_test = np.load('../blending/DecomposableAttention_Feat_1125_f1_test_0.8622.npy')

    infersent_train = np.load('../blending/InferSent_f1_train_0.8572.npy')
    infersent_test = np.load('../blending/InferSent_f1_test_0.8572.npy')

    infersent_feat_train = np.load('../blending/InferSent_Feat_f1_train_0.8642.npy')
    infersent_feat_test = np.load('../blending/InferSent_Feat_f1_test_0.8642.npy')

    siamese_textcnn_train = np.load('../blending/SiameseTextCNN_1124_f1_train_0.8531.npy')
    siamese_textcnn_test = np.load('../blending/SiameseTextCNN_1124_f1_test_0.8531.npy')

    siamese_textcnn_feat_train = np.load('../blending/SiameseTextCNN_Feat_1124_f1_train_0.8570.npy')
    siamese_textcnn_feat_test = np.load('../blending/SiameseTextCNN_Feat_1124_f1_test_0.8570.npy')

    train_pred = np.vstack(
        (decatt_train, infersent_train, infersent_feat_train, siamese_textcnn_train,
         siamese_textcnn_feat_train)).T
    test_pred = np.vstack(
        (decatt_test, infersent_test, infersent_feat_test, siamese_textcnn_test,
         siamese_textcnn_feat_test)).T

    train_feat = pd.read_csv('../data/simplified_train_feat.csv').values
    test_feat = pd.read_csv('../data/simplified_test_feat.csv').values

    train_x = np.concatenate((train_pred, train_feat), axis=1)
    test_x = np.concatenate((test_pred, test_feat), axis=1)

    return train_x, truth, test_x


def stacking_layer1_oof_pred(model, model_name, train_data, train_label, test_x, num_fold, layer=1):
    fold_len = train_data.shape[0] // num_fold
    skf_indices = []
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    for i, (train_idx, valid_idx) in enumerate(skf.split(np.ones(train_data.shape[0]), train_label)):
        skf_indices.extend(valid_idx.tolist())
    train_pred = np.zeros(train_data.shape[0])
    test_pred = np.zeros(test_x.shape[0])
    for fold in range(num_fold):
        print(f'Processing fold {fold}...')
        fold_start = fold * fold_len
        fold_end = (fold + 1) * fold_len
        if fold == num_fold - 1:
            fold_end = train_data.shape[0]
        train_indices = skf_indices[:fold_start] + skf_indices[fold_end:]
        test_indices = skf_indices[fold_start:fold_end]
        train_x = train_data[train_indices]
        train_y = train_label[train_indices]
        cv_test_x = train_data[test_indices]
        model.fit(train_x, train_y)

        pred = model.predict_proba(cv_test_x)[:, 1]
        train_pred[test_indices] = pred
        pred = model.predict_proba(test_x)[:, 1]
        test_pred += pred / num_fold

    # y_pred = np.argmax(train_pred, axis=1)
    y_pred = np.round(train_pred)
    score = f1_score(train_label, y_pred, average='macro')
    print(score)

    pred_dir = f'../stacking/{model_name}/{model.random_state}/'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    train_path = pred_dir + f'train_{score:.6f}'
    test_path = pred_dir + f'test_{score:.6f}'

    np.save(train_path, train_pred)
    np.save(test_path, test_pred)


if __name__ == '__main__':

    train_data, train_label, test_data = get_layer1_input()

    for i in range(1, 11):
        model = LGBMClassifier(num_leaves=15, learning_rate=0.05, n_estimators=50000, subsample=0.8,
                               colsample_bytree=0.8, random_state=i)
        stacking_layer1_oof_pred(model, f'lgbm_15leaves_50k', train_data, train_label, test_data, 10)

    # shallow lgbm
    # for i in range(1, 11):
    #     model = LGBMClassifier(num_leaves=7, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8,
    #                            random_state=i)
    #     stacking_layer1_oof_pred(model, f'lgbm_7leaves_1125_full', train_data, train_label, test_data, 10)

    # medium lgbm
    # for i in range(1, 11):
    #     model = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8,
    #                            random_state=i)
    #     stacking_layer1_oof_pred(model, f'lgbm_31leaves_1125_full', train_data, train_label, test_data, 10)

    # deep lgbm
    # for i in range(1, 11):
    #     model = LGBMClassifier(num_leaves=127, learning_rate=0.05, n_estimators=500, subsample=0.8,
    #                            colsample_bytree=0.8, random_state=i)
    #     stacking_layer1_oof_pred(model, f'lgbm_127leaves_1125_full', train_data, train_label, test_data, 10)
