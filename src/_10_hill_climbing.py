from collections import Counter

from _00_imports import *


def faster_hill_climbing_ensemble(model_names, train_preds, test_preds, n_iter=20, file_name=None):
    """

    Run Hill climbing ensemble to find sub-optimal weights for models.

    :param model_names: list of model names
    :param train_preds: list of out-of-fold predictions of different models on training set
    :param test_preds: list of out-of-fold predictions of different models on test set
    :param n_iter: number of iteration
    :param file_name: save blending result if file_name is provided.
    :rtype: None
    """
    # y_true = np.load('../../data/label.npy')
    y_true = truth

    pred_indices = []
    # rt = []
    best_f1 = -1
    best_pred_indices = None

    for iter in range(n_iter):
        best_pred_index = -1
        current_best_f1 = -1

        # Select a best prediction
        for i, pred in enumerate(train_preds):
            pred_indices.append(i)

            coefs = Counter(pred_indices)
            y_pred = np.zeros(len(y_true))
            total = len(pred_indices)
            for idx in coefs.keys():
                y_pred += (coefs[idx] / total) * train_preds[idx]
            # y_pred = y_pred.argmax(axis=1)
            y_pred = np.round(y_pred)
            f1 = f1_score(y_true, y_pred)

            if f1 > current_best_f1:
                best_pred_index = i
                current_best_f1 = f1

            pred_indices.pop(-1)

        pred_indices.append(best_pred_index)

        if current_best_f1 > best_f1:
            best_f1 = current_best_f1
            best_pred_indices = pred_indices.copy()

        print(f'Epoch {iter}: {current_best_f1:.6f}')

    counter = Counter(best_pred_indices)
    total = len(best_pred_indices)
    for i, count in counter.items():
        print(f'{model_names[i]} : {count/total:.4f}')

    ensemble_train = np.zeros_like(train_preds[0])
    ensemble_test = np.zeros_like(test_preds[0])

    for i in range(len(model_names)):
        ensemble_train += train_preds[i] * (counter[i] / total)
        ensemble_test += test_preds[i] * (counter[i] / total)

    score = f1_score(y_true, np.round(ensemble_train))
    print(f'Hill climb blending of {len(model_names)} models: f1_macro {score:.6f}')

    if file_name is not None:
        np.save(f'../stacking/{file_name}_train_{score:.6f}', ensemble_train)
        np.save(f'../stacking/{file_name}_test_{score:.6f}', ensemble_test)


if __name__ == '__main__':
    model_names = [f'lgbm_{num_leaves}leaves_{seed}' for num_leaves in [7, 31, 127] for seed in range(1, 11)]
    train_preds, test_preds = [], []

    # for num_leaves in [7, 31, 127]:
    #     for seed in range(1, 11):
    #         for name in os.listdir(f'../stacking/lgbm_{num_leaves}leaves/{seed}'):
    #             if 'train' in name:
    #                 train_preds.append(np.load(f'../stacking/lgbm_{num_leaves}leaves/{seed}/{name}'))
    #             elif 'test' in name:
    #                 test_preds.append(np.load(f'../stacking/lgbm_{num_leaves}leaves/{seed}/{name}'))

    for num_leaves in [7, 31, 127]:
        for seed in range(1, 11):
            for name in os.listdir(f'../stacking/lgbm_{num_leaves}leaves_1125/{seed}'):
                if 'train' in name:
                    train_preds.append(np.load(f'../stacking/lgbm_{num_leaves}leaves_1125/{seed}/{name}'))
                elif 'test' in name:
                    test_preds.append(np.load(f'../stacking/lgbm_{num_leaves}leaves_1125/{seed}/{name}'))

    faster_hill_climbing_ensemble(model_names, train_preds, test_preds, n_iter=100,
                                  file_name='reproduce_all_depth_merged_lgbm_1125')
