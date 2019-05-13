import torch
import torch.nn as nn
import os
import re
import random
from textblob import TextBlob
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
from textblob import TextBlob

from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from model_config import model_name, attn_model, hidden_size
from model_config import encoder_n_layers, decoder_n_layers, dropout, batch_size
from model_config import device, loadFilename, checkpoint_iter
from model_config import save_dir, corpus_name
from model_config import embedding_size, loadFilename
from train_config import clip, learning_rate, decoder_learning_ratio, n_iteration
from train_config import print_every, save_every
from voc import Voc, normalizeString
from squad_loader import ANSS_TAG, ANSE_TAG
from evaluate import GreedySearchDecoder, evaluate

#loadFile = None
'''
loadFile = os.path.join(save_dir, model_name, corpus_name,
                        '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                        '{}_checkpoint.tar'.format(checkpoint_iter))
'''

voc = Voc(corpus_name)
#load model if a loadFile is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, embedding_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(embedding_size, hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, embedding_size, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

searcher = GreedySearchDecoder(encoder, decoder)

encoder.eval()
decoder.eval()

nlp = en_core_web_sm.load()

def generate_questions(input_text):
    qa_pairs = []
    sentences = input_text.split('.')
    input_indexes = random.sample(range(0, len(sentences) - 1), len(sentences) // 2)
    #for index in input_indexes:
    for index in range(0, len(sentences) - 1):
        qa_pair = qg_from_sentence(sentences[index])
        if qa_pair:
            qa_pairs.append(qa_pair)
    return qa_pairs

def qg_from_sentence(input_sentence):
    input_sentence = normalizeString(input_sentence)
    doc = nlp(input_sentence)
    answer_candidates = []
    '''
    for entry in doc.ents:
        answer_candidates.append(entry.text)
    '''
    sent_blob = TextBlob(input_sentence)
    answer_candidates = sent_blob.noun_phrases
    if len(answer_candidates) == 0:
        return None
    answer = random.choice(answer_candidates)
    tagged_answer = ANSS_TAG + " " + answer + " " + ANSE_TAG
    input_sentence = input_sentence.replace(answer, tagged_answer)
    print(input_sentence)
    print(answer)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    question = ' '.join(output_words)
    question_sentences = question.split('?')
    question = question_sentences[0]
    question = question + '?'
    return (question, answer)

