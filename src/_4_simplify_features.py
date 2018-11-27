from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from _0_imports import *

# Use LGBM to select the most important 100 features
if __name__ == '__main__':
    train_feat = np.load('../data/feat_train.npy')
    test_feat = np.load('../data/feat_test.npy')

    train_input = np.load('../data/feat_train.npy')
    lgbm = LGBMClassifier(n_estimators=5000)
    train_x, test_x, train_y, test_y = train_test_split(train_input, truth, test_size=0.2, random_state=1)

    lgbm.fit(train_x, train_y)
    pred = lgbm.predict(test_x)
    print(f1_score(test_y, pred))
    indices = np.argsort(lgbm.feature_importances_)
    importance = lgbm.feature_importances_
    top_indices = indices[-100:][::-1]

    train_feat_df = pd.read_csv('../data/feat_train.csv')
    test_feat_df = pd.read_csv('../data/feat_test.csv')
    feat_names = train_feat_df.columns.values
    selected_feat_names = feat_names[top_indices]

    train_feat_df[selected_feat_names].to_csv('../data/simplified_train_feat.csv', index=False)
    test_feat_df[selected_feat_names].to_csv('../data/simplified_test_feat.csv', index=False)
    np.save('../data/simplified_train_feat', train_feat_df[selected_feat_names].values)
    np.save('../data/simplified_test_feat', test_feat_df[selected_feat_names].values)
