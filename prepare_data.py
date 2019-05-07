import torch
import itertools
from voc import PAD_token, EOS_token, UNK_token, normalizeString
from squad_loader import ANSS_TAG, ANSE_TAG

def indexesFromSentence(voc, sentence):
    #return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    indexes = []
    for word in sentence.split(' '):
        if word in voc.word2index:
            indexes.append(voc.word2index[word])
        else:
            indexes.append(UNK_token)
    indexes.append(EOS_token)
    return indexes

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns mask for tagging answers
def ansMask(l):
    ret = []
    for sentence in l:
        ret.append(ansSentMask(sentence))
    return ret

def ansSentMask(sentence):
    mask = []
    ans_flag = False
    for word in sentence.split(' '):
        if not ans_flag:
            if word != ANSS_TAG:
                mask.append(0)
            else:
                ans_flag = True
        else:
            if word != ANSE_TAG:
                mask.append(1)
            else:
                ans_flag = False
    # Append 0 for EOS tag
    mask.append(0)
    return mask


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    ansMaskList = ansMask(l)
    no_ans_l = []
    for sentence in l:
        s1 = sentence.replace(ANSS_TAG, '')
        s2 = s1.replace(ANSE_TAG, '')
        s3 = normalizeString(s2)
        no_ans_l.append(s3)
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in no_ans_l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)

    answerMaskList = zeroPadding(ansMaskList)
    answerMask = torch.FloatTensor(answerMaskList)
    return padVar, lengths, answerMask

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths, answerMask = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len, answerMask
