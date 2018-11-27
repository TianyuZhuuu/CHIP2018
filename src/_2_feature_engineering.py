from _0_imports import *

question_df = pd.read_csv('../input/question_id.csv')
question_df = question_df.set_index('qid')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
concat_df = pd.concat((train_df, test_df))
features = pd.DataFrame()

# Add wid & cid to train & test df
concat_df['q1_wid'] = concat_df['qid1'].apply(lambda qid: question_df.loc[qid]['wid'])
concat_df['q2_wid'] = concat_df['qid2'].apply(lambda qid: question_df.loc[qid]['wid'])
concat_df['q1_cid'] = concat_df['qid1'].apply(lambda qid: question_df.loc[qid]['cid'])
concat_df['q2_cid'] = concat_df['qid2'].apply(lambda qid: question_df.loc[qid]['cid'])
concat_df = concat_df[['q1_wid', 'q2_wid', 'q1_cid', 'q2_cid']]


def cosine(m1, m2):
    return np.einsum('ij,ij->i', m1, m2) / (np.linalg.norm(m1, axis=1) * np.linalg.norm(m2, axis=1))


# ===========================================================================
# Statistic features: token count, unique token count, unique token ratio...
def statistic_features(row, granularity='w'):
    if granularity == 'w':
        q1, q2 = row.q1_wid, row.q2_wid
    else:
        q1, q2 = row.q1_cid, row.q2_cid

    q1_tokens = q1.split()
    q2_tokens = q2.split()
    q1_unique_tokens = set(q1_tokens)
    q2_unique_tokens = set(q2_tokens)
    shared_tokens = q1_unique_tokens.intersection(q2_unique_tokens)

    q1_token_count = len(q1_tokens)
    q2_token_count = len(q2_tokens)
    q1_unique_token_count = len(q1_unique_tokens)
    q2_unique_token_count = len(q2_unique_tokens)
    q1_unique_ratio = q1_unique_token_count / q1_token_count
    q2_unique_ratio = q2_unique_token_count / q2_token_count

    min_token_count = min(q1_token_count, q2_token_count)
    max_token_count = max(q1_token_count, q2_token_count)
    min_unique_token_count = min(q1_unique_token_count, q2_unique_token_count)
    max_unique_token_count = max(q1_unique_token_count, q2_unique_token_count)
    min_unique_ratio = min(q1_unique_ratio, q2_unique_ratio)
    max_unique_ratio = max(q1_unique_ratio, q2_unique_ratio)
    count_add = q1_token_count + q2_token_count
    count_sub = abs(q1_token_count - q2_token_count)
    count_mul = q1_token_count * q2_token_count
    unique_count_add = q1_unique_token_count + q2_unique_token_count
    unique_count_sub = abs(q1_unique_token_count - q2_unique_token_count)
    unique_count_mul = q1_unique_token_count * q2_unique_token_count
    unique_ratio_add = q1_unique_ratio + q2_unique_ratio
    unique_ratio_sub = abs(q1_unique_ratio - q2_unique_ratio)
    unique_ratio_mul = q1_unique_ratio * q2_unique_ratio
    shared_ratio_min = len(shared_tokens) / max(q1_unique_token_count, q2_unique_token_count)
    shared_ratio_max = len(shared_tokens) / min(q1_unique_token_count, q2_unique_token_count)

    rt = [min_token_count, max_token_count,
          min_unique_token_count, max_unique_token_count,
          min_unique_ratio, max_unique_ratio,
          count_add, count_sub, count_mul,
          unique_count_add, unique_count_sub, unique_count_mul,
          unique_ratio_add, unique_ratio_sub, unique_ratio_mul,
          shared_ratio_min, shared_ratio_max]

    return pd.Series(rt)


# ===========================================================================

# ===========================================================================
# Vector Space Model features: Count & TF-IDF: l1-distance, l2-distance, L-infinity-distance
word_count_vectorizer = CountVectorizer()
word_tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)
char_count_vectorizer = CountVectorizer()
char_tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)

word_sequences = question_df['wid'].values
char_sequences = question_df['cid'].values

word_count_vectorizer.fit(word_sequences)
word_tfidf_vectorizer.fit(word_sequences)
char_count_vectorizer.fit(char_sequences)
char_tfidf_vectorizer.fit(char_sequences)


def vector_space_features():
    data_dict = {}
    # Word features
    q1_wid, q2_wid = concat_df['q1_wid'].values, concat_df['q2_wid'].values
    q1_count_vecs = word_count_vectorizer.transform(q1_wid).toarray()
    q2_count_vecs = word_count_vectorizer.transform(q2_wid).toarray()
    count_diff_vecs = q1_count_vecs - q2_count_vecs
    # Count Vectorizer based similarity & distance measures
    data_dict['word_count_cos_sim'] = cosine(q1_count_vecs, q2_count_vecs)
    data_dict['word_count_l1'] = np.linalg.norm(count_diff_vecs, ord=1, axis=1)
    data_dict['word_count_l2'] = np.linalg.norm(count_diff_vecs, ord=2, axis=1)
    data_dict['word_count_linf'] = np.linalg.norm(count_diff_vecs, ord=np.inf, axis=1)
    # Tfidf Vectorizer based similarity & distance measures
    q1_tfidf_vecs = word_tfidf_vectorizer.transform(q1_wid).toarray()
    q2_tfidf_vecs = word_tfidf_vectorizer.transform(q2_wid).toarray()
    tfidf_diff_vecs = q1_tfidf_vecs - q2_tfidf_vecs
    data_dict['word_tfidf_cos_sim'] = cosine(q1_tfidf_vecs, q2_tfidf_vecs)
    data_dict['word_tfidf_l1'] = np.linalg.norm(tfidf_diff_vecs, ord=1, axis=1)
    data_dict['word_tfidf_l2'] = np.linalg.norm(tfidf_diff_vecs, ord=2, axis=1)
    data_dict['word_tfidf_linf'] = np.linalg.norm(tfidf_diff_vecs, ord=np.inf, axis=1)
    # Char features
    q1_cid, q2_cid = concat_df['q1_cid'].values, concat_df['q2_cid'].values
    q1_count_vecs = char_count_vectorizer.transform(q1_cid).toarray()
    q2_count_vecs = char_count_vectorizer.transform(q2_cid).toarray()
    count_diff_vecs = q1_count_vecs - q2_count_vecs
    # Count Vectorizer based similarity & distance measures
    data_dict['char_count_cos_sim'] = cosine(q1_count_vecs, q2_count_vecs)
    data_dict['char_count_l1'] = np.linalg.norm(count_diff_vecs, ord=1, axis=1)
    data_dict['char_count_l2'] = np.linalg.norm(count_diff_vecs, ord=2, axis=1)
    data_dict['char_count_linf'] = np.linalg.norm(count_diff_vecs, ord=np.inf, axis=1)
    # Tfidf Vectorizer based similarity & distance measures
    q1_tfidf_vecs = char_tfidf_vectorizer.transform(q1_cid).toarray()
    q2_tfidf_vecs = char_tfidf_vectorizer.transform(q2_cid).toarray()
    tfidf_diff_vecs = q1_tfidf_vecs - q2_tfidf_vecs
    data_dict['char_tfidf_cos_sim'] = cosine(q1_tfidf_vecs, q2_tfidf_vecs)
    data_dict['char_tfidf_l1'] = np.linalg.norm(tfidf_diff_vecs, ord=1, axis=1)
    data_dict['char_tfidf_l2'] = np.linalg.norm(tfidf_diff_vecs, ord=2, axis=1)
    data_dict['char_tfidf_linf'] = np.linalg.norm(tfidf_diff_vecs, ord=np.inf, axis=1)
    return pd.DataFrame(data_dict)


# ===========================================================================

# ===========================================================================
# Fuzzy Wuzzy features: calculate the differences between sequences
def fuzzywuzzy_features(row, granularity='w'):
    if granularity == 'w':
        q1, q2 = row.q1_wid, row.q2_wid
    else:
        q1, q2 = row.q1_cid, row.q2_cid
    ratio = fuzz.ratio(q1, q2)
    partial_ratio = fuzz.partial_ratio(q1, q2)
    token_set_ratio = fuzz.token_set_ratio(q1, q2)
    token_sort_ratio = fuzz.token_sort_ratio(q1, q2)

    rt = [ratio, partial_ratio, token_set_ratio, token_sort_ratio]
    return pd.Series(rt)


# ===========================================================================

# ===========================================================================
# Word Vector features: use IDF to calculate the weighted average of word vectors then calculate cosine, l1, l2 and word mover distance
def word_vector_features():
    from torchtext import data

    def word_mover_distance(model, row, granularity='w'):
        if granularity == 'w':
            q1, q2 = row.q1_wid, row.q2_wid
            return model.wmdistance(q1.split(), q2.split())
        else:
            q1, q2 = row.q1_cid, row.q2_cid
            return model.wmdistance(q1.split(), q2.split())

    # Word Mover Distance
    filepath = f'../data/word_vectors.txt'
    tmppath = f'../data/gensim_tmp_word_vector.txt'
    if not os.path.exists(tmppath):
        glove2word2vec(filepath, tmppath)
    word_model = KeyedVectors.load_word2vec_format(tmppath)

    filepath = f'../data/char_vectors.txt'
    tmppath = f'../data/gensim_tmp_char_vector.txt'
    if not os.path.exists(tmppath):
        glove2word2vec(filepath, tmppath)
    char_model = KeyedVectors.load_word2vec_format(tmppath)

    word_wmd = [word_mover_distance(word_model, row, 'w') for row in concat_df.itertuples(index=False)]
    char_wmd = [word_mover_distance(char_model, row, 'c') for row in concat_df.itertuples(index=False)]

    # tf-idf weighted word vector as sentence representation
    # then calculate cosine similarity, l1-norm, l2-norm
    word_embedding_path = '../data/word_vectors.txt'
    char_embedding_path = '../data/char_vectors.txt'
    cache = '../cache'
    word_vectors = Vectors(word_embedding_path, cache)
    char_vectors = Vectors(char_embedding_path, cache)
    word_vectors.unk_init = lambda x: init.uniform_(x, -0.05, 0.05)
    char_vectors.unk_init = lambda x: init.uniform_(x, -0.05, 0.05)
    wordTEXT = data.Field(batch_first=True)
    charTEXT = data.Field(batch_first=True)

    fields = [('q1_word', wordTEXT),
              ('q2_word', wordTEXT),
              ('q1_char', charTEXT),
              ('q2_char', charTEXT)]
    examples = [data.Example.fromlist(row, fields) for row in
                concat_df.itertuples(index=False)]
    dataset = data.Dataset(examples, fields)

    wordTEXT.build_vocab(dataset, min_freq=1, vectors=word_vectors)
    charTEXT.build_vocab(dataset, min_freq=1, vectors=char_vectors)

    word_embedding = wordTEXT.vocab.vectors
    char_embedding = charTEXT.vocab.vectors

    num_word = word_embedding.size(0)
    num_char = char_embedding.size(0)

    word_index2idf = np.zeros(num_word)
    char_index2idf = np.zeros(num_char)

    word_counter = Counter()
    char_counter = Counter()
    for wid in question_df['wid']:
        word_counter.update(wid.split())
    for cid in question_df['cid']:
        char_counter.update(cid.split())

    N = len(concat_df)

    # 0 --> <unk>
    # 1 --> <pad>
    # start from 2
    for i in range(2, num_word):
        word = wordTEXT.vocab.itos[i]
        idf = np.log(N / word_counter[word])
        word_index2idf[i] = idf

    for i in range(2, num_char):
        char = charTEXT.vocab.itos[i]
        idf = np.log(N / char_counter[char])
        char_index2idf[i] = idf

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    word_idf = torch.tensor(word_index2idf, dtype=torch.float32).to(device)
    char_idf = torch.tensor(char_index2idf, dtype=torch.float32).to(device)

    word_embedder = nn.Embedding.from_pretrained(word_embedding).to(device)
    char_embedder = nn.Embedding.from_pretrained(char_embedding).to(device)

    word_similarity, char_similarity = [], []
    word_l1, char_l1 = [], []
    word_l2, char_l2 = [], []

    iter = data.BucketIterator(dataset, 1024, sort_key=None, shuffle=False, device=torch.device('cuda:0'),
                               sort_within_batch=False)

    for data in iter:
        # [batch, seq_len]
        q1_word, q2_word, q1_char, q2_char = data.q1_word, data.q2_word, data.q1_char, data.q2_char

        q1_word_embed = word_embedder(q1_word)  # [batch, seq_len, 300]
        q2_word_embed = word_embedder(q2_word)
        q1_char_embed = char_embedder(q1_char)  # [batch, seq_len, 300]
        q2_char_embed = char_embedder(q2_char)

        batch = q1_word_embed.size(0)

        q1_word_flat = q1_word.view(-1)  # [batch * seq_len]
        q2_word_flat = q2_word.view(-1)
        q1_char_flat = q1_char.view(-1)
        q2_char_flat = q2_char.view(-1)

        q1_word_idfs = word_idf.index_select(0, index=q1_word_flat).view(batch, -1)  # [batch, seq_len]
        q2_word_idfs = word_idf.index_select(0, index=q2_word_flat).view(batch, -1)
        q1_char_idfs = char_idf.index_select(0, index=q1_char_flat).view(batch, -1)
        q2_char_idfs = char_idf.index_select(0, index=q2_char_flat).view(batch, -1)

        # q1_word_idfs = F.softmax(q1_word_idfs, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        # q2_word_idfs = F.softmax(q2_word_idfs, dim=1).unsqueeze(-1)
        # q1_char_idfs = F.softmax(q1_char_idfs, dim=1).unsqueeze(-1)
        # q2_char_idfs = F.softmax(q2_char_idfs, dim=1).unsqueeze(-1)

        q1_word_idfs = (q1_word_idfs / q1_word_idfs.sum(dim=1, keepdim=True)).unsqueeze(-1)  # [batch, seq_len, 1]
        q2_word_idfs = (q2_word_idfs / q2_word_idfs.sum(dim=1, keepdim=True)).unsqueeze(-1)
        q1_char_idfs = (q1_char_idfs / q1_char_idfs.sum(dim=1, keepdim=True)).unsqueeze(-1)
        q2_char_idfs = (q2_char_idfs / q2_char_idfs.sum(dim=1, keepdim=True)).unsqueeze(-1)

        q1_word_repre = torch.bmm(q1_word_embed.transpose(1, 2), q1_word_idfs).squeeze()  # [batch, 300]
        q2_word_repre = torch.bmm(q2_word_embed.transpose(1, 2), q2_word_idfs).squeeze()
        q1_char_repre = torch.bmm(q1_char_embed.transpose(1, 2), q1_char_idfs).squeeze()
        q2_char_repre = torch.bmm(q2_char_embed.transpose(1, 2), q2_char_idfs).squeeze()

        word_cos_sim = F.cosine_similarity(q1_word_repre, q2_word_repre, dim=1)
        char_cos_sim = F.cosine_similarity(q1_char_repre, q2_char_repre, dim=1)
        word_l1_norm = torch.norm(q1_word_repre - q2_word_repre, p=1, dim=-1)
        char_l1_norm = torch.norm(q1_char_repre - q2_char_repre, p=1, dim=-1)
        word_l2_norm = torch.norm(q1_word_repre - q2_word_repre, p=2, dim=-1)
        char_l2_norm = torch.norm(q1_char_repre - q2_char_repre, p=2, dim=-1)

        word_similarity.append(word_cos_sim)
        char_similarity.append(char_cos_sim)
        word_l1.append(word_l1_norm)
        char_l1.append(char_l1_norm)
        word_l2.append(word_l2_norm)
        char_l2.append(char_l2_norm)

    word_similarity = torch.cat(word_similarity).cpu().numpy()
    char_similarity = torch.cat(char_similarity).cpu().numpy()
    word_l1 = torch.cat(word_l1).cpu().numpy()
    char_l1 = torch.cat(char_l1).cpu().numpy()
    word_l2 = torch.cat(word_l2).cpu().numpy()
    char_l2 = torch.cat(char_l2).cpu().numpy()

    rt = pd.DataFrame({'word_wmd': word_wmd, 'word_wv_cos_sim': word_similarity,
                       'word_wv_l1': word_l1, 'word_wv_l2': word_l2,
                       'char_wmd': char_wmd, 'char_wv_cos_sim': char_similarity,
                       'char_wv_l1': char_l1, 'char_wv_l2': char_l2})
    return rt


# ===========================================================================

# ===========================================================================
# Topic Model features: perform dimension reduction, the mesure the cosine, l1-distance, l2-distance, linf-distance between them
names = ['LDA', 'LSI', 'NMF']
dimensions = list(range(10, 151, 10))

word_count_vecs = word_count_vectorizer.fit_transform(word_sequences)
char_count_vecs = char_count_vectorizer.fit_transform(char_sequences)
word_tfidf_vecs = word_tfidf_vectorizer.fit_transform(word_sequences)
char_tfidf_vecs = char_tfidf_vectorizer.fit_transform(char_sequences)

word_count_ldas = [LatentDirichletAllocation(n_components=dim, random_state=2018) for dim in dimensions]
word_count_lsas = [TruncatedSVD(n_components=dim, random_state=2018) for dim in dimensions]
word_count_nmfs = [NMF(n_components=dim, random_state=2018) for dim in dimensions]

word_tfidf_ldas = [LatentDirichletAllocation(n_components=dim, random_state=2018) for dim in dimensions]
word_tfidf_lsas = [TruncatedSVD(n_components=dim, random_state=2018) for dim in dimensions]
word_tfidf_nmfs = [NMF(n_components=dim, random_state=2018) for dim in dimensions]

word_count_vectors = word_count_vectorizer.transform(word_sequences)
word_tfidf_vectors = word_tfidf_vectorizer.transform(word_sequences)
char_count_vectors = char_count_vectorizer.transform(char_sequences)
char_tfidf_vectors = char_tfidf_vectorizer.transform(char_sequences)

for i, models in enumerate([word_count_ldas, word_count_lsas, word_count_nmfs]):
    for j, model in enumerate(models):
        model.fit(word_count_vectors)
        print(f'Word & CountVectorizer & {names[i]}-{dimensions[j]}d fitted..')

for i, models in enumerate([word_tfidf_ldas, word_tfidf_lsas, word_tfidf_nmfs]):
    for j, model in enumerate(models):
        model.fit(word_tfidf_vectors)
        print(f'Word & TfidfVectorizer & {names[i]}-{dimensions[j]}d fitted..')

char_count_ldas = [LatentDirichletAllocation(n_components=dim, random_state=2018) for dim in dimensions]
char_count_lsas = [TruncatedSVD(n_components=dim, random_state=2018) for dim in dimensions]
char_count_nmfs = [NMF(n_components=dim, random_state=2018) for dim in dimensions]

char_tfidf_ldas = [LatentDirichletAllocation(n_components=dim, random_state=2018) for dim in dimensions]
char_tfidf_lsas = [TruncatedSVD(n_components=dim, random_state=2018) for dim in dimensions]
char_tfidf_nmfs = [NMF(n_components=dim, random_state=2018) for dim in dimensions]

for i, models in enumerate([char_count_ldas, char_count_lsas, char_count_nmfs]):
    for j, model in enumerate(models):
        model.fit(char_count_vectors)
        print(f'Char & CountVectorizer & {names[i]}-{dimensions[j]}d fitted..')

for i, models in enumerate([char_tfidf_ldas, char_tfidf_lsas, char_tfidf_nmfs]):
    for j, model in enumerate(models):
        model.fit(char_tfidf_vectors)
        print(f'Char & TfidfVectorizer & {names[i]}-{dimensions[j]}d fitted..')


def topic_model_features():
    # rt = [0.0] * (len(names) * len(dimensions) * 4 * 2)
    data_dict = {}
    # count_vectorizer, tfidf_vectorizer = word_count_vectorizer, word_tfidf_vectorizer
    q1, q2 = concat_df['q1_wid'], concat_df['q2_wid']
    q1_count_vecs, q2_count_vecs = word_count_vectorizer.transform(q1), word_count_vectorizer.transform(q2)
    q1_tfidf_vecs, q2_tfidf_vecs = word_tfidf_vectorizer.transform(q1), word_tfidf_vectorizer.transform(q2)
    for i, models in enumerate([word_count_ldas, word_count_lsas, word_count_nmfs]):
        for j, model in enumerate(models):
            prefix = f'word_count_{names[i]}_{dimensions[j]}d'
            v1, v2 = model.transform(q1_count_vecs), model.transform(q2_count_vecs)
            data_dict[f'{prefix}_cos_sim'] = cosine(v1, v2)
            data_dict[f'{prefix}_l1'] = np.linalg.norm(v1 - v2, ord=1, axis=1)
            data_dict[f'{prefix}_l2'] = np.linalg.norm(v1 - v2, ord=2, axis=1)
            data_dict[f'{prefix}_linf'] = np.linalg.norm(v1 - v2, ord=np.inf, axis=1)
    for i, models in enumerate([word_tfidf_ldas, word_tfidf_lsas, word_tfidf_nmfs]):
        for j, model in enumerate(models):
            prefix = f'word_tfidf_{names[i]}_{dimensions[j]}d'
            v1, v2 = model.transform(q1_tfidf_vecs), model.transform(q2_tfidf_vecs)
            data_dict[f'{prefix}_cos_sim'] = cosine(v1, v2)
            data_dict[f'{prefix}_l1'] = np.linalg.norm(v1 - v2, ord=1, axis=1)
            data_dict[f'{prefix}_l2'] = np.linalg.norm(v1 - v2, ord=2, axis=1)
            data_dict[f'{prefix}_linf'] = np.linalg.norm(v1 - v2, ord=np.inf, axis=1)
    q1, q2 = concat_df['q1_cid'], concat_df['q2_cid']
    q1_count_vecs, q2_count_vecs = char_count_vectorizer.transform(q1), char_count_vectorizer.transform(q2)
    q1_tfidf_vecs, q2_tfidf_vecs = char_tfidf_vectorizer.transform(q1), char_tfidf_vectorizer.transform(q2)
    for i, models in enumerate([char_count_ldas, char_count_lsas, char_count_nmfs]):
        for j, model in enumerate(models):
            prefix = f'char_count_{names[i]}_{dimensions[j]}d'
            v1, v2 = model.transform(q1_count_vecs), model.transform(q2_count_vecs)
            data_dict[f'{prefix}_cos_sim'] = cosine(v1, v2)
            data_dict[f'{prefix}_l1'] = np.linalg.norm(v1 - v2, ord=1, axis=1)
            data_dict[f'{prefix}_l2'] = np.linalg.norm(v1 - v2, ord=2, axis=1)
            data_dict[f'{prefix}_linf'] = np.linalg.norm(v1 - v2, ord=np.inf, axis=1)
    for i, models in enumerate([char_tfidf_ldas, char_tfidf_lsas, char_tfidf_nmfs]):
        for j, model in enumerate(models):
            prefix = f'char_count_{names[i]}_{dimensions[j]}d'
            v1, v2 = model.transform(q1_tfidf_vecs), model.transform(q2_tfidf_vecs)
            data_dict[f'{prefix}_cos_sim'] = cosine(v1, v2)
            data_dict[f'{prefix}_l1'] = np.linalg.norm(v1 - v2, ord=1, axis=1)
            data_dict[f'{prefix}_l2'] = np.linalg.norm(v1 - v2, ord=2, axis=1)
            data_dict[f'{prefix}_linf'] = np.linalg.norm(v1 - v2, ord=np.inf, axis=1)

    return pd.DataFrame(data_dict)


# ===========================================================================

# ===========================================================================
N = len(question_df)

word_counter, char_counter = Counter(), Counter()
for q in question_df['wid']:
    word_counter.update(q.split())
for q in question_df['cid']:
    char_counter.update(q.split())

word2idf = {word: np.log(N / count) for word, count in word_counter.items()}
char2idf = {char: np.log(N / count) for char, count in char_counter.items()}


def prefix_suffix_features(row, granularity='w'):
    # Word features
    if granularity == 'w':
        q1, q2 = row.q1_wid, row.q2_wid
        token2idf = word2idf
    else:
        q1, q2 = row.q1_cid, row.q2_cid
        token2idf = char2idf

    q1_tokens, q2_tokens = q1.split(), q2.split()
    q1_tokens_reversed, q2_tokens_reversed = q1_tokens[::-1], q2_tokens[::-1]

    max_score = max(len(q1_tokens), len(q2_tokens))
    max_recip_score = sum([1 / pos for pos in range(1, max_score + 1)])
    max_idf_score = max(sum([token2idf[w] for w in q1_tokens]), sum([token2idf[w] for w in q2_tokens]))

    prefix_score, recip_prefix_score, idf_prefix_score = 0.0, 0.0, 0.0
    for i, (tok1, tok2) in enumerate(zip(q1_tokens, q2_tokens)):
        if tok1 == tok2:
            prefix_score += 1.0
            recip_prefix_score += 1 / (i + 1)
            idf_prefix_score += token2idf[tok1]
        else:
            prefix_score /= max_score
            recip_prefix_score /= max_recip_score
            idf_prefix_score /= max_idf_score

    suffix_score, recip_suffix_score, idf_suffix_score = 0.0, 0.0, 0.0
    for i, (tok1, tok2) in enumerate(zip(q1_tokens_reversed, q2_tokens_reversed)):
        if tok1 == tok2:
            suffix_score += 1.0
            recip_suffix_score += 1 / (i + 1)
            idf_suffix_score += token2idf[tok1]
        else:
            suffix_score /= max_score
            recip_suffix_score /= max_recip_score
            idf_suffix_score /= max_idf_score

    rt = [prefix_score, recip_prefix_score, idf_prefix_score,
          suffix_score, recip_suffix_score, idf_suffix_score]
    return pd.Series(rt)


if __name__ == '__main__':
    question_df = pd.read_csv('../input/question_id.csv')
    question_df = question_df.set_index('qid')

    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    concat_df = pd.concat((train_df, test_df))
    features = pd.DataFrame()

    # Add wid & cid to train & test df
    concat_df['q1_wid'] = concat_df['qid1'].apply(lambda qid: question_df.loc[qid]['wid'])
    concat_df['q2_wid'] = concat_df['qid2'].apply(lambda qid: question_df.loc[qid]['wid'])
    concat_df['q1_cid'] = concat_df['qid1'].apply(lambda qid: question_df.loc[qid]['cid'])
    concat_df['q2_cid'] = concat_df['qid2'].apply(lambda qid: question_df.loc[qid]['cid'])
    concat_df = concat_df[['q1_wid', 'q2_wid', 'q1_cid', 'q2_cid']]

    # Statistic features
    feat_names = ['{}_min_count', '{}_max_count',
                  '{}_min_unique_count', '{}_max_unique_count',
                  '{}_min_unique_ratio', '{}_max_unique_ratio',
                  '{}_count_add', '{}_count_sub', '{}_count_mul',
                  '{}_unique_count_add', '{}_unique_count_sub', '{}_unique_count_mul',
                  '{}_unique_ratio_add', '{}_unique_ratio_sub', '{}_unique_ratio_mul',
                  '{}_shared_ratio_min', '{}_shared_ratio_max']
    statistic_feat_df = pd.DataFrame()
    statistic_feat_df[[feat.format('word') for feat in feat_names]] = concat_df.apply(
        lambda row: statistic_features(row, 'w'),
        axis=1)
    statistic_feat_df[[feat.format('char') for feat in feat_names]] = concat_df.apply(
        lambda row: statistic_features(row, 'c'),
        axis=1)
    statistic_feat_df[:len(train_df)].to_csv('../data/statistic_feat_train.csv', index=False)
    statistic_feat_df[len(train_df):].to_csv('../data/statistic_feat_test.csv', index=False)
    print('Statistic features generated..')

    # Vector Space Model features
    vsm_feat_df = vector_space_features()
    vsm_feat_df[:len(train_df)].to_csv('../data/vsm_feat_train.csv', index=False)
    vsm_feat_df[len(train_df):].to_csv('../data/vsm_feat_test.csv', index=False)
    features = pd.concat((features, vsm_feat_df), axis=1)
    print('Vector Space Model features generated..')

    # FuzzyWuzzy features
    feat_names = ['{}_fuzz_ratio', '{}_fuzz_partial_ratio', '{}_fuzz_token_set_ratio', '{}_fuzz_token_sort_ratio']
    fuzz_feat_df = pd.DataFrame()
    fuzz_feat_df[[feat.format('word') for feat in feat_names]] = concat_df.apply(
        lambda row: fuzzywuzzy_features(row, 'w'),
        axis=1)
    fuzz_feat_df[[feat.format('char') for feat in feat_names]] = concat_df.apply(
        lambda row: fuzzywuzzy_features(row, 'c'),
        axis=1)
    fuzz_feat_df[:len(train_df)].to_csv('../data/fuzz_feat_train.csv', index=False)
    fuzz_feat_df[len(train_df):].to_csv('../data/fuzz_feat_test.csv', index=False)
    print('Fuzzy Wuzzy features generated..')

    feat_names = ['{}_prefix_score', '{}_recip_prefix_score', '{}_idf_prefix_score',
                  '{}_suffix_score', '{}_recip_suffix_score', '{}_idf_suffix_score']
    prefix_suffix_feat_df = pd.DataFrame()
    prefix_suffix_feat_df[[feat.format('word') for feat in feat_names]] = concat_df.apply(
        lambda row: prefix_suffix_features(row, granularity='w'), axis=1
    )
    prefix_suffix_feat_df[[feat.format('char') for feat in feat_names]] = concat_df.apply(
        lambda row: prefix_suffix_features(row, granularity='c'), axis=1
    )
    prefix_suffix_feat_df[:len(train_df)].to_csv('../data/prefix_suffix_feat_train.csv', index=False)
    prefix_suffix_feat_df[len(train_df):].to_csv('../data/prefix_suffix_feat_test.csv', index=False)
    print('Prefix Suffix features generated..')

    # Word Vector features
    wv_feat_df = word_vector_features()
    wv_feat_df[:len(train_df)].to_csv('../data/word_vector_feat_train.csv', index=False)
    wv_feat_df[len(train_df):].to_csv('../data/word_vector_feat_test.csv', index=False)
    features = pd.concat((features, wv_feat_df), axis=1)
    print('Word Vector features generated..')

    # Topic Modelling features
    topic_modelling_feat_df = topic_model_features()
    topic_model_feat_df = pd.concat((features, topic_modelling_feat_df), axis=1)
    topic_modelling_feat_df[:len(train_df)].to_csv('../data/topic_model_feat_train.csv', index=False)
    topic_modelling_feat_df[len(train_df):].to_csv('../data/topic_model_feat_test.csv', index=False)
    print('Topic Modelling fetures generated..')
