from _0_imports import *

if __name__ == '__main__':

    # No post-processing can be applied since there's no overlap between questions appeared
    # in the training set and test set. No question appeared both in training set and test
    # set.

    question_df = pd.read_csv('../input/question_id.csv')
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')

    train_qid = set()
    test_qid = set()

    for row in train_df.itertuples(index=False):
        qid1, qid2 = row.qid1, row.qid2
        train_qid.add(qid1)
        train_qid.add(qid2)

    for row in test_df.itertuples(index=False):
        qid1, qid2 = row.qid1, row.qid2
        test_qid.add(qid1)
        test_qid.add(qid2)

    intersect = train_qid.intersection(test_qid)
    print(f'# of intersection: {len(intersect)}')  # 0
    # No intersection between training & test set
