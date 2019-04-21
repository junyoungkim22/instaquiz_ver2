import simplejson as json
import spacy
import sys
import random
import os

'''
This file is taken and modified from
https://github.com/deepakkumar1984/QANet2/blob/master/prepro.py
'''

nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def process_file(file_name):
    squad_loc = os.path.join("data", "squad", file_name)
    with open(squad_loc, "r") as data_file:
        source = json.load(data_file)
        ret = []
        print("Processing SquAD dataset...")
        for article in source["data"]:
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                #context_tokens = word_tokenize(context)
                context_qas = []
                for qa in para["qas"]:
                    question = qa["question"].replace("''", '" ').replace("``", '" ')
                    ans_txt_pos = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        ans_txt_pos.append((answer_text, (answer_start, answer_end)))
                    context_qas.append((question, ans_txt_pos))
                ret.append((context, context_qas))
        print("Processing done!")
        return ret

def prepare_par_pairs():
    pairs = []
    data = process_file("train-v2.0.json")
    for context, context_qas in data:
        for question, answers in context_qas:
            for ans_txt, (answer_start, answer_end) in answers:
                pairs.append((context, question))
    return pairs

def prepare_sent_pairs():
    pairs = prepare_ans_sent_pairs()
    new_pairs = []
    for pair in pairs:
        context, question = pair
        new_context = context.replace('<anss>', '')
        new_context = new_context.replace('<anse>', '')
        new_pairs.append((new_context, question))
    return new_pairs

def prepare_ans_tagged_pairs():
    pairs = []
    data = process_file("train-v2.0.json")
    for context, context_qas in data:
        for question, answers in context_qas:
            for ans_txt, (answer_start, answer_end) in answers:
                tagged_context = context[:answer_end] + " <anse> " + context[answer_end:]
                tagged_context = tagged_context[:answer_start] + " <anss> " + tagged_context[answer_start:]
                pairs.append((tagged_context, question))
    return pairs

def prepare_ans_sent_pairs():
    pairs = prepare_ans_tagged_pairs()
    new_pairs = []
    for pair in pairs:
        context, question = pair
        new_sentences = []
        for sent in context.split('.'):
            if "<anss>" in sent:
                new_sentences.append(sent)
            elif "<anse>" in sent:
                new_sentences.append(sent)
        new_context = ' '.join(new_sentences)
        new_pairs.append((new_context, question))
    return new_pairs
                

def ans_tag_test():
    data = process_file("train-v2.0.json")
    ans_tag_pairs = prepare_ans_tagged_pairs(data)
    for i in range(10):
        pair = random.choice(ans_tag_pairs)
        context, question = pair
        print(context)
        print("&"*80)
        print(question)
        print("*"*80)

def print_pairs(pairs):
    for i in range(50):
        pair = random.choice(pairs)
        context, question = pair
        print(context)
        print('&'*80)
        print(question)
        print('*'*80)

#ans_tag_test()

def test():
    data = process_file("train-v2.0.json")
    i = 0
    for context, context_qas in data:
        print(context)
        print('*'*80)
        for question, answers in context_qas:
            print(question)
            print('-'*80)
            for txt, (start, end) in answers:
                print(txt)
                print(start)
                print(end)
                print('&'*80)
        print("\n")
        i += 1
        if(i == 20):
            break

def dev_test():
    data = process_file("dev-v2.0.json")
    i = 0
    for context, context_qas in data:
        print(context)
        print('*'*80)
        for question, answers in context_qas:
            print(question)
            print('-'*80)
            for txt, (start, end) in answers:
                print(txt)
                print(start)
                print(end)
                print('&'*80)
        print("\n")
        i += 1
        if(i == 20):
            break
