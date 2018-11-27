from _0_imports import *


def blending(base_dir, mean='mean', save=True, f1_or_loss='both'):
    if mean == 'mean':
        mean_f = lambda preds: np.mean(preds, axis=0)
    elif mean == 'gmean':
        mean_f = gmean
    else:
        raise ValueError(f'Unsupported mean function : {mean}')

    train_pred, test_pred = [], []
    for seed in os.listdir(base_dir):
        for name in os.listdir(f'{base_dir}/{seed}'):
            temp = np.load(f'{base_dir}/{seed}/{name}')
            if 'f1' in name:
                if f1_or_loss == 'both' or f1_or_loss == 'f1':
                    if 'train' in name:
                        train_pred.append(temp)
                    else:
                        test_pred.append(temp)
            elif 'loss' in name:
                if f1_or_loss == 'both' or f1_or_loss == 'loss':
                    if 'train' in name:
                        train_pred.append(temp)
                    else:
                        test_pred.append(temp)

    train_pred, test_pred = mean_f(train_pred), mean_f(test_pred)
    if save:
        f1 = f1_score(label, np.round(train_pred))
        np.save(f"{base_dir.replace('oof', 'blending')}_{f1_or_loss}_train_{f1:.4f}", train_pred)
        np.save(f"{base_dir.replace('oof', 'blending')}_{f1_or_loss}_test_{f1:.4f}", test_pred)
    return train_pred, test_pred


if __name__ == '__main__':
    # blending('../oof/InferSent')
    # blending('../oof/InferSent_Feat', f1_or_loss='both')
    # blending('../oof/SiameseTextCNN_1124', f1_or_loss='loss')
    # blending('../oof/SiameseTextCNN_Feat_1124', f1_or_loss='both')
    # blending('../oof/DecomposableAttention_1125', f1_or_loss='loss')
    blending('../oof/DecomposableAttention_Feat_1125', f1_or_loss='loss')
    # blending('../oof/DecomposableAttention')
    # blending('../oof/1DCNN')
    pass
