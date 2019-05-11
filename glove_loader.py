import pickle
import bcolz
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from global_var import MAPPER

glove_path = 'glove'
glove_dim = '200'


def process_glove_file():
    vectors = bcolz.carray(np.zeros(1), rootdir=glove_path +  '/6B.' + glove_dim + '.dat', mode='w')
    words = []
    idx = 0
    word2idx = {}
    with open(glove_path +  '/glove.6B.' + glove_dim + 'd.txt', 'rb') as f:
        i = 0
        for l in tqdm(f):
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            i += 1
            if i == 400000:
                break

        
    vectors = bcolz.carray(vectors[1:].reshape((400000, int(glove_dim))), rootdir=glove_path + '/6B.' + glove_dim + '.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(glove_path +  '/6B.' + glove_dim + '_words.pkl', 'wb'))
    pickle.dump(word2idx, open(glove_path +  '/6B.' + glove_dim + '_idx.pkl', 'wb'))

def create_glove_vect_dict():
    vectors = bcolz.open(glove_path + '/6B.' + glove_dim + '.dat')[:]
    words = pickle.load(open(glove_path +  '/6B.' + glove_dim + '_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path +  '/6B.' + glove_dim + '_idx.pkl', 'rb'))
    return {w: vectors[word2idx[w]] for w in words}

def make_weights_matrix(emb_dim):
    matrix_len = MAPPER.n_words
    weights_matrix = np.zeros(shape=(matrix_len, emb_dim))
    words_found = 0
    glove = create_glove_vect_dict()
    index2word = MAPPER.index2word
    for i in range(matrix_len):
        try:
            word = index2word[i]
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    return torch.from_numpy(weights_matrix)

def create_emb_layer(emb_dim, non_trainable=False):
    weights_matrix = make_weights_matrix(emb_dim)
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight_requires_grad = False
    return emb_layer
