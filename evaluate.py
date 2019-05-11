import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu

from prepare_data import indexesFromSentence, ansSentMask
from voc import normalizeString
from voc import MAX_LENGTH, SOS_token
from model_config import device
from squad_loader import ANSS_TAG, ANSE_TAG

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, answerMask):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length, answerMask)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # Get answer mask
    ans_mask_list_batch = [ansSentMask(sentence)]
    # Remove answer tags
    sentence = sentence.replace(ANSS_TAG, ' ')
    sentence = sentence.replace(ANSE_TAG, ' ')
    sentence = normalizeString(sentence)
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    ans_mask_input_batch = torch.FloatTensor(ans_mask_list_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    ans_mask_input_batch = ans_mask_input_batch.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length, ans_mask_input_batch)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = ' '.join(output_words)
            output_sentence = output_sentence.split('?')[0] + '?'
            #print('Bot:', ' '.join(output_words))
            print(output_sentence)

        except KeyError:
            print("Error: Encountered unknown word.")

def dev_evaluate(encoder, decoder, dev_pairs, searcher, voc):
    encoder.eval()
    decoder.eval()
    bleu_scores = []
    for pair in dev_pairs:
        text, ans_question = pair
        text = normalizeString(text)
        output_words = evaluate(encoder, decoder, searcher, voc, text)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        gen_question = ' '.join(output_words)
        gen_question = gen_question.split('?')[0] + '?'
        bleu_score = sentence_bleu([normalizeString(ans_question).split(' ')], gen_question.split(' '), weights=[1])
        '''
        print(normalizeString(ans_question))
        print(gen_question)
        print(bleu_score)
        print('*'*70)
        '''
        bleu_scores.append(bleu_score)
    print("BLEU score average: %f" % (sum(bleu_scores) / len(bleu_scores)))
    encoder.train()
    decoder.train()
