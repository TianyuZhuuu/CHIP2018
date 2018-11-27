import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors


class CHIPDataset():
    def __init__(self, qid_path, train_path, test_path, word_path, char_path, num_folds=10, batch_size=32,
                 seed=2018):
        question_df = pd.read_csv(qid_path)
        question_df = question_df.set_index('qid')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.num_folds = num_folds
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train_df['q1_wid'] = train_df['qid1'].apply(lambda qid: question_df.loc[qid]['wid'])
        train_df['q2_wid'] = train_df['qid2'].apply(lambda qid: question_df.loc[qid]['wid'])
        train_df['q1_cid'] = train_df['qid1'].apply(lambda qid: question_df.loc[qid]['cid'])
        train_df['q2_cid'] = train_df['qid2'].apply(lambda qid: question_df.loc[qid]['cid'])
        self.train_df = train_df[['q1_wid', 'q2_wid', 'q1_cid', 'q2_cid', 'label']]

        test_df['q1_wid'] = test_df['qid1'].apply(lambda qid: question_df.loc[qid]['wid'])
        test_df['q2_wid'] = test_df['qid2'].apply(lambda qid: question_df.loc[qid]['wid'])
        test_df['q1_cid'] = test_df['qid1'].apply(lambda qid: question_df.loc[qid]['cid'])
        test_df['q2_cid'] = test_df['qid2'].apply(lambda qid: question_df.loc[qid]['cid'])
        self.test_df = test_df[['q1_wid', 'q2_wid', 'q1_cid', 'q2_cid']]

        self.word_embedding_path = word_path
        self.char_embedding_path = char_path

        cache = '../cache'
        if not os.path.exists(cache):
            os.mkdir(cache)

        self.word_vectors = Vectors(self.word_embedding_path, cache)
        self.char_vectors = Vectors(self.char_embedding_path, cache)
        self.word_vectors.unk_init = lambda x: init.uniform_(x, -0.05, 0.05)
        self.char_vectors.unk_init = lambda x: init.uniform_(x, -0.05, 0.05)
        self.wordTEXT = data.Field(batch_first=True)
        self.charTEXT = data.Field(batch_first=True)
        self.LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        train_dataset = self.generate_dataset()
        test_dataset = self.generate_dataset(role='test')
        self.wordTEXT.build_vocab(train_dataset, test_dataset, min_freq=1, vectors=self.word_vectors)
        self.charTEXT.build_vocab(train_dataset, test_dataset, min_freq=1, vectors=self.char_vectors)
        self.word_embedding = self.wordTEXT.vocab.vectors
        self.char_embedding = self.charTEXT.vocab.vectors

    def generate_dataset(self, indices=None, role='train'):
        if role == 'train' or role == 'valid':
            fields = [('index', data.Field(sequential=False, use_vocab=False, dtype=torch.long)),
                      ('q1_word', self.wordTEXT),
                      ('q2_word', self.wordTEXT),
                      ('q1_char', self.charTEXT),
                      ('q2_char', self.charTEXT),
                      ('label', self.LABEL)]
            if indices is not None:
                examples = [data.Example.fromlist(row, fields) for row in
                            self.train_df.iloc[indices, :].itertuples()]
            else:
                examples = [data.Example.fromlist(row, fields) for row in
                            self.train_df.itertuples()]
        elif role == 'test':
            fields = [('index', data.Field(sequential=False, use_vocab=False, dtype=torch.long)),
                      ('q1_word', self.wordTEXT),
                      ('q2_word', self.wordTEXT),
                      ('q1_char', self.charTEXT),
                      ('q2_char', self.charTEXT)]
            examples = [data.Example.fromlist(row, fields) for row in self.test_df.itertuples()]
        else:
            raise ValueError(f'Role must be \'train\', \'valid\' or \'test\'')
        return data.Dataset(examples, fields)

    def generate_iterator_fold(self, fold=0):
        skf = StratifiedKFold(self.num_folds, True, self.seed)
        label = self.train_df['label'].astype(float)
        train_indices, valid_indices = list(skf.split(np.zeros_like(label), label))[fold]
        # Build train/valid/test dataset
        train_dataset = self.generate_dataset(train_indices, role='train')
        valid_dataset = self.generate_dataset(valid_indices, role='valid')
        test_dataset = self.generate_dataset(role='test')

        # Create iterator
        train_iter = data.BucketIterator(train_dataset, self.batch_size,
                                         sort_key=lambda ex: len(ex.q1_word) + len(ex.q2_word) + \
                                                             len(ex.q1_char) + len(ex.q2_char),
                                         shuffle=True, device=self.device, sort=True, train=True)
        valid_iter = data.BucketIterator(valid_dataset, self.batch_size, sort_key=None,
                                         device=self.device, shuffle=False, sort_within_batch=False)
        test_iter = data.BucketIterator(test_dataset, self.batch_size, sort_key=None,
                                        device=self.device, shuffle=False, sort_within_batch=False)

        return train_iter, valid_iter, test_iter

    def reorder_oof_prediction(self, prediction):
        indices = []
        pred = np.zeros_like(prediction)
        skf = StratifiedKFold(self.num_folds, True, self.seed)
        label = self.train_df['label'].astype(float)
        for train_indices, valid_indices in skf.split(np.zeros_like(label), label):
            indices.extend(valid_indices)
        pred[indices] = prediction
        return pred
