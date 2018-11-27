from sklearn.preprocessing import normalize, MinMaxScaler

from _0_imports import *

if __name__ == '__main__':
    # Save all features' name
    columns = []

    fuzz_train = pd.read_csv('../data/fuzz_feat_train.csv')
    fuzz_test = pd.read_csv('../data/fuzz_feat_test.csv')
    columns.extend(fuzz_train.columns.values)

    prefix_suffix_train = pd.read_csv('../data/prefix_suffix_feat_train.csv')
    prefix_suffix_test = pd.read_csv('../data/prefix_suffix_feat_test.csv')
    columns.extend(prefix_suffix_train.columns.values)

    statistic_train = pd.read_csv('../data/statistic_feat_train.csv')
    statistic_test = pd.read_csv('../data/statistic_feat_test.csv')
    columns.extend(statistic_train.columns.values)

    topic_model_train = pd.read_csv('../data/topic_model_feat_train.csv')
    topic_model_test = pd.read_csv('../data/topic_model_feat_test.csv')
    columns.extend(topic_model_train.columns.values)

    vsm_train = pd.read_csv('../data/vsm_feat_train.csv')
    vsm_test = pd.read_csv('../data/vsm_feat_test.csv')
    columns.extend(vsm_train.columns.values)

    word_vector_train = pd.read_csv('../data/vsm_feat_train.csv')
    word_vector_test = pd.read_csv('../data/vsm_feat_test.csv')
    columns.extend(word_vector_train.columns.values)

    feat_train = pd.concat(
        [fuzz_train, prefix_suffix_train, statistic_train, topic_model_train, vsm_train, word_vector_train],
        axis=1).values
    feat_test = pd.concat(
        [fuzz_test, prefix_suffix_test, statistic_test, topic_model_test, vsm_test, word_vector_test],
        axis=1).values

    feat_concat = np.vstack((feat_train, feat_test))
    feat_concat = np.nan_to_num(feat_concat)

    # Normalize the features
    scaler = MinMaxScaler()
    scaler.fit(feat_concat)
    feat_train_normalized = scaler.transform(feat_concat[:20000])
    feat_test_normalized = scaler.transform(feat_concat[20000:])

    # Save in numpy format
    np.save('../data/feat_train', feat_train_normalized)
    np.save('../data/feat_test', feat_test_normalized)

    # Save in csv format
    new_train_df = pd.DataFrame(feat_train_normalized, columns=columns)
    new_test_df = pd.DataFrame(feat_test_normalized, columns=columns)
    new_train_df.to_csv('../data/feat_train.csv', index=False)
    new_test_df.to_csv('../data/feat_test.csv', index=False)
