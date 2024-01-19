# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:01:27 2023

@author: ariel
"""

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from datasets import load_dataset
import torch
import random
from transformers import BartForConditionalGeneration
from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import math
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
#%% INITIALISE MODELS
#initialise summarization model
model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

#initialise sentiment analysis model
sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
sentiment_config = AutoConfig.from_pretrained(sentiment_model)
# PT
sentiment_model_classifier = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
#%% DISTILIBART INITALISATION
# model_ckpt = "sshleifer/distilbart-cnn-6-6"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = BartForConditionalGeneration.from_pretrained(model_ckpt)
#%% DATASET
dataset = load_dataset("multi_news")
data = dataset['train']['document']
#%% UTIL FUNCTIONS
def generate_sample(data, size=4):
    condition=True
    while condition:
        #pick random sample from list of data
        sample = random.choice(data)
        while sample.count('|||||') != size-1:
            sample = random.choice(data)
        #separate sample into versions
        sample_list = sample.split("|||||")
        #check if article contains enough content
        if len(max(sample_list)) >= 500:
            condition=False
    print('Number of articles (sample size): %s' % (len(sample_list)))
    return sample, sample_list

def tokenize_inputs(sample_list, max_length=256):
    #encode each article independently, to remove context sharing between them
    inputs = []
    for sample in sample_list:
        sample_input_id = tokenizer(sample, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        inputs.append(sample_input_id)
        
    return inputs

def combine_inputs(inputs):
    input_ids_list = []
    attention_mask_list = []
    for text in inputs:
        input_ids_list.append(text['input_ids'])
        attention_mask_list.append(text['attention_mask'])
    
    input_ids = torch.cat(input_ids_list, dim=1)
    attention_mask = torch.cat(attention_mask_list, dim=1)
    return input_ids, attention_mask
#%% PRE-PROCESSS INPUTS
sample, sample_list = generate_sample(data, size=4)
inputs = tokenize_inputs(sample_list, max_length=512)
input_ids, attention_mask = combine_inputs(inputs)
#%% GENERATE SUMMARIES

#test summary for one article  --uncomment to test
# summaries = model.generate(input_ids=inputs[0]['input_ids'], attention_mask=inputs[0]['attention_mask'], min_length=100, max_length=256, do_sample=True, top_p=0.92, top_k=0, num_return_sequences=3)
# decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
# print(decoded_summaries)
summaries = model.generate(input_ids=inputs[0]['input_ids'], attention_mask=inputs[0]['attention_mask'],
                           min_new_tokens=100, max_new_tokens=256, do_sample=True, temperature=0.87, top_k=10)
                           #return_dict_in_generate=True, output_scores=True)
decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
print(decoded_summaries[0])

#summary for combine encoding
# sum_list=[]
# for i in range(3):
#     summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask,min_length=250, max_length=256, do_sample=True, top_k=10, temperature = 0.95, top_p=0.9) #top_p introduces some randomness
#     print(summaries.size())
#     decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
#     sum_list.append(decoded_summaries)
#%%
transition_scores = model.compute_transition_scores(summaries.sequences, summaries.scores, summaries.beam_indices, normalize_logits=True)
#prob=np.sum(np.exp(transition_scores.numpy()))
prob =  torch.exp(transition_scores.sum(axis=1))
print(prob)
#%%

def iterated_learning(text, pop_count=100, gen_count=4, sample_count=4, k=10, temperature=0.95, max_new_tokens = 200, min_new_tokens = 120): #max_length=200, min_length=120):
    #CREATE GENERATED TEXT DICTIONARY
    gen_dict = {gen: {} for gen in range(gen_count)}
    
    #PRE-PROCESS
    inputs = tokenize_inputs(text, max_length=max_new_tokens)
    input_ids, attention_mask = combine_inputs(inputs)
    
    #gen_summaries = []
    for gen in range(gen_count):
        pop_summaries = []
        for pop in range(pop_count):
            summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask, min_new_tokens = min_new_tokens, max_new_tokens=max_new_tokens, do_sample=True, top_k=k, temperature=temperature)
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries][0]
            pop_summaries.append(decoded_summaries)
        #gen_summaries.append(pop_summaries)  
        
        gen_dict[gen]['raw_text'] = pop_summaries
        #randomly sample from population summaries
        summ = random.choices(pop_summaries, k=sample_count)
        inputs = tokenize_inputs(summ, max_length=max_new_tokens)
        input_ids, attention_mask = combine_inputs(inputs)
    
    return gen_dict #gen_summaries

def get_sentiment(text):
    """

    Parameters
    ----------
    text : n-List or STR
        DESCRIPTION.

    Returns
    -------
    scores : ndarray
        n array of scores arrays or flat array [negative, neutral, positive].

    """
    scores = []
    for t in text:
        encoded_input = sentiment_tokenizer(t, return_tensors='pt')
        output = sentiment_model_classifier(**encoded_input)
        scores.append(softmax(output[0][0].detach().numpy()))
    #scores = softmax(scores)
    
    if type(text) == str:
        return scores[0]
    else:
        return scores
            
    
def analyse_sentiment(gen_dict):
    sent_dict = {gen: {'negative': {}, 'neutral': {}, 'positive': {}} for gen in gen_dict.keys()} #gen_dict.copy()
    for gen in gen_dict.keys():
        #scores = get_sentiment(generated_text[gen])
        scores = get_sentiment(gen_dict[gen]['raw_text'])
        neg_bin = [score[0] for score in scores]
        neut_bin = [score[1] for score in scores]
        pos_bin = [score[2] for score in scores]
        
        sent_dict[gen]['scores'] = scores
        sent_dict[gen]['label'] = [score.argmax() for score in scores]
        sent_dict[gen]['negative']['raw'] = neg_bin
        sent_dict[gen]['neutral']['raw'] = neut_bin
        sent_dict[gen]['positive']['raw'] = pos_bin
        
        #avg sentiment
        sent_dict[gen]['negative']['avg'] = np.average(neg_bin)
        sent_dict[gen]['neutral']['avg'] = np.average(neut_bin)
        sent_dict[gen]['positive']['avg'] = np.average(pos_bin)
        #cumulative sentiment
        sent_dict[gen]['negative']['cum'] = np.sum(neg_bin)
        sent_dict[gen]['neutral']['cum'] = np.sum(neut_bin)
        sent_dict[gen]['positive']['cum'] = np.sum(pos_bin)
        
    return sent_dict

def plot_sentiment_ovt(sent_dict):
    plt.plot(range(len(sent_dict)), [sent_dict[gen]['negative']['avg'] for gen in sent_dict.keys()], label = 'negative', color = 'red')
    plt.plot(range(len(sent_dict)), [sent_dict[gen]['neutral']['avg'] for gen in sent_dict.keys()], label = 'neutral', color = 'blue')
    plt.plot(range(len(sent_dict)), [sent_dict[gen]['positive']['avg'] for gen in sent_dict.keys()], label = 'positive', color = 'green')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Softmax Sentiment Score')
    plt.show()
    
def sentiment_distribution(sent_dict, gen_count):
    if np.sqrt(gen_count)-round(np.sqrt(gen_count)) < 0:
        nrow = math.ceil(np.sqrt(gen_count))
        ncol = nrow
        #if np.sqrt(pop_count)-round(np.sqrt(pop_count)) >= 0:
    else:
        nrow = math.floor(np.sqrt(gen_count))
        ncol = math.ceil(np.sqrt(gen_count))
        
    fig, axs = plt.subplots(nrow, ncol)
    axs = axs.flatten()
    #number of samples in each label in each generation
    #average distribution in each generation
    
    for gen in sent_dict.keys():
        axs[gen].bar([0,1,2], [sent_dict[gen]['label'].count(0),
                               sent_dict[gen]['label'].count(1),
                               sent_dict[gen]['label'].count(2)],
                     tick_label = ['negative', 'neutral', 'positive'])
        
    plt.show()
    
#%% PARAMS
pop_count = 5
gen_count = 2
#%%
_, source_list = generate_sample(data, size=4)
#%%
generated_summaries= iterated_learning(source_list, pop_count=10, gen_count = 15, sample_count=4, temperature=0.90)

#%% EMBED TEXT
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#%%
sample, sample_list = generate_sample(data, size=2)
#%%
embeddings = model.encode(sample_list[0])
print(embeddings)

#%%
sentiment_dict = analyse_sentiment(generated_summaries)
#%%
plot_sentiment_ovt(sentiment_dict)
sentiment_distribution(sentiment_dict, 6)