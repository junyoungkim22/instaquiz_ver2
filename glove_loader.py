import numpy as np
from io import open

def make_weights(embed_size, glove_path, voc):
    print("Making weights...")
    vocab_size = voc.num_words
    sd = 1/np.sqrt(embed_size)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)
    with open(glove_path, encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = voc.word2index.get(word, None)
            if id is not None:
                weights[id] = np.array(line[1:], dtype=np.float32)
    print("Done!")
    return weights

