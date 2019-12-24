#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
import pandas as pd
from seq2seq_model import Embedding_Seq2SeqSummarizer
import numpy as np
import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
import time
import re
import json
from rouge import Rouge


def main():
    smoothie = SmoothingFunction().method4
    data_dir_path = 'data'
    model_dir_path = 'models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/lenta_test.csv")
    X = df['text']
    Y = df['title']

    # loading our model
    model_path = Embedding_Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)
    with open(model_path, 'rb') as data:
        config = pickle.load(data)
    
    summarizer = Embedding_Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Embedding_Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))
    

    print('start predicting ...')
    result = ''
    bleus = []
    beam_bleus = []
    rouge = Rouge()
    refs, greedy_hyps, beam_hyps = [], [], [] 
    
    # some decent examples
    demo = [3, 5, 31, 36, 37, 47, 54, 55, 99, 19, 39, 119] 

    for i in demo:
    # for i in range(50):
        x = X[i]
        actual_headline = Y[i]
        refs.append(actual_headline)

        headline = summarizer.summarize(x)
        greedy_hyps.append(headline)

        beam_headline = summarizer.beam_search(x, 3)
        beam_hyps.append(beam_headline)

        bleu = sentence_bleu([word_tokenize(actual_headline.lower())], word_tokenize(headline), smoothing_function=smoothie)
        bleus.append(bleu)
        beam_bleu = sentence_bleu([word_tokenize(actual_headline.lower())], word_tokenize(beam_headline), smoothing_function=smoothie)
        beam_bleus.append(beam_bleu)

        # if i % 200 == 0 and i != 0:
        #         print(i)
        #         print("BLEU: ", np.mean(np.array(bleus)))
        #         print("BEAM BLEU: ", np.mean(np.array(beam_bleus)))

        print(f'№ {i}')
        # print('Article: ', x)       
        print('Original Headline: ', actual_headline)        
        print('Generated Greedy Headline: ', headline)
        print('Generated Beam Headline: ', beam_headline)
        print('\n')

    print ('__________METRICS SUMMARY____________')
    avg_greedy_scores = rouge.get_scores(greedy_hyps, refs, avg=True)
    rouge1f = avg_greedy_scores['rouge-1']['f']
    rouge2f = avg_greedy_scores['rouge-2']['f']
    rougelf = avg_greedy_scores['rouge-l']['f']
    score = np.mean([rouge1f, rouge2f, rougelf])
    print('Greedy Rouge (Dialogue 2019): ', score)
    avg_beam_scores = rouge.get_scores(beam_hyps, refs, avg=True)


    rouge1f = avg_beam_scores['rouge-1']['f']
    rouge2f = avg_beam_scores['rouge-2']['f']
    rougelf = avg_beam_scores['rouge-l']['f']
    score = np.mean([rouge1f, rouge2f, rougelf])
    print('Beam search Rouge (Dialogue 2019): ', score)
         

    def average(lst): 
        return float(sum(lst)) / float(len(lst)) 

    print("Greedy Bleu: ", average(bleus))
    print("Beam search Bleu: ", average(beam_bleus))
    print ('_____________________________________')

if __name__ == '__main__':
    main()
