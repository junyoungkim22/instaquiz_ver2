import torch
import os

model_name = 'elmo_model0'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
embedding_size = 300
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
save_dir = os.path.join("data", "save")
corpus_name = "squad"

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 16000
#loadFilename = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}-{}'.format(encoder_n_layers, decoder_n_layers, embedding_size, hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter))

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

use_glove = False
unk_replace = False
use_elmo = True
