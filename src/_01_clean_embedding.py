import numpy as np

# TorchText use space as delimiter to load word vectors
# the provided word vectors are seperated by \t
# the function is to transform the provided word vectors into standard form
if __name__ == '__main__':
    word_vectors = []
    with open('../input/word_embedding.txt') as f:
        for line in f:
            word_vectors.append(line.replace('\t', ' '))
    with open('../data/word_vectors.txt', 'w') as f:
        for line in word_vectors:
            f.write(line)
    print('Word vectors cleaned..')

    char_vectors = []
    with open('../input/char_embedding.txt') as f:
        for line in f:
            char_vectors.append(line.replace('\t', ' '))
    with open('../data/char_vectors.txt', 'w') as f:
        for line in char_vectors:
            f.write(line)
    print('Char vectors cleaned..')
