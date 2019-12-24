from __future__ import print_function
from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import pickle

###### MODEL PARAMETERS #######
HIDDEN_UNITS = 256
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10
###############################

class Deep_Seq2SeqSummarizer():

    model_name = 'deep_Seq2Seq_enoder_decoder_emb'

    def __init__(self, config):
        """Unpacking config that we obtained from news_loader"""

        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        encoder_inputs = Input(shape=(None,), name='encoder_inputs') 
        encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_input_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm1', return_sequences=True)
        encoder_outputs, encoder_state_h1, encoder_state_c1 = encoder_lstm(encoder_embedding(encoder_inputs))
        _, encoder_state_h2, encoder_state_c2 = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm2')(encoder_outputs)
        encoder_states = [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2] # don't need this (or change in decoder)
        

        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(input_dim=self.num_target_tokens, output_dim=HIDDEN_UNITS,
                                      name='decoder_embedding')

        decoder_lstm1 = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm1')
        decoder_outputs1, decoder_state_h1, decoder_state_c1 = decoder_lstm1(decoder_embedding(decoder_inputs),
                                                                             initial_state=[encoder_state_h1, encoder_state_c1])
        decoder_lstm2 = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm2')
        decoder_outputs2, decoder_state_h2, decoder_state_c2 = decoder_lstm1(decoder_embedding(decoder_inputs),
                                                                             initial_state=[encoder_state_h2, encoder_state_c2])
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs2)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model
        
        # Inference
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h1 = Input(shape=(HIDDEN_UNITS,))
        decoder_state_input_c1 = Input(shape=(HIDDEN_UNITS,))
        decoder_state_input_h2 = Input(shape=(HIDDEN_UNITS,))
        decoder_state_input_c2 = Input(shape=(HIDDEN_UNITS,))
        decoder_state_inputs = [decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]
        

        decoder_outputs, state_h1, state_c1 = decoder_lstm1(decoder_embedding(decoder_inputs), initial_state=decoder_state_inputs[:2])
        decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_state_inputs[-2:])
        

        decoder_states = [state_h1, state_c1, state_h2, state_c2]


        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)


    def transform_input_text(self, texts):
        temp = []
        for text in texts:
            x = []
            for word in text.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for text in texts:
            x = []
            line2 = 'START ' + text.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length))
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx] = w2idx
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Deep_Seq2SeqSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Deep_Seq2SeqSummarizer.model_name + '-config.pickle'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Deep_Seq2SeqSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        
        # Saving
        config_file_path = Deep_Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Deep_Seq2SeqSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        # np.save(config_file_path, self.config)
        # Save config as a pickle and load it predict
        with open(config_file_path, 'wb') as data:
            pickle.dump(self.config, data, protocol=pickle.HIGHEST_PROTOCOL)



        architecture_file_path = Deep_Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1 
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)

        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1)) 
        target_seq[0, 0] = self.target_word2idx['START']

        target_text = ''
        target_text_len = 0
        terminated = False

        while not terminated:
            output_tokens, h1, c1, h2, c2 = self.decoder_model.predict([target_seq] + states_value)
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sample_token_idx

            states_value = [h1, c1, h2, c2]
        return target_text.strip()

