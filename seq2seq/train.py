from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from plot_utils import plot_and_save_history
from seq2seq_model import Embedding_Seq2SeqSummarizer
from news_loader import fit_text


LOAD_EXISTING_WEIGHTS = False
TRAINING_EPOCHS = 12


def main():
    data_dir_path = 'data'
    report_dir_path = 'reports'
    model_dir_path = 'models'
    
    print('loading data ...')
    df = pd.read_csv(data_dir_path + 'merged_ria_lenta.csv')

    print('extract configuration from input texts ...')
    Y = df['title']
    X = df['text']
    
    config = fit_text(X, Y)

    summarizer = Embedding_Seq2SeqSummarizer(config)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

#     print('demo size: ', len(Xtrain))
#     print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=TRAINING_EPOCHS)
    history_plot_file_path = report_dir_path + '/' + Embedding_Seq2SeqSummarizer.model_name + '-history.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()