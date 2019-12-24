from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
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
DEFAULT_BATCH_SIZE = 256
VERBOSE = 1
DEFAULT_EPOCHS = 12
###############################

class Embedding_Seq2SeqSummarizer():

    model_name = 'Embedding_Seq2Seq_enoder_decoder_emb'

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

        # Encoder
        encoder_inputs = Input(shape=(None,), name='encoder_inputs') 
        encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_input_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(input_dim=self.num_target_tokens, output_dim=HIDDEN_UNITS,
                                      name='decoder_embedding')

        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embedding(decoder_inputs),
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)


        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model
        
        # Inference
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(HIDDEN_UNITS,))
        decoder_state_input_c = Input(shape=(HIDDEN_UNITS,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
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
                # for embedddings we use different shape 
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length))
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx] = w2idx # this is changed for embeddings
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    # Saving
    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Embedding_Seq2SeqSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Embedding_Seq2SeqSummarizer.model_name + '-config.pickle'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Embedding_Seq2SeqSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version

        config_file_path = Embedding_Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Embedding_Seq2SeqSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)

        # Here we will save config as a pickle and load it predict
        with open(config_file_path, 'wb') as data:
            pickle.dump(self.config, data, protocol=pickle.HIGHEST_PROTOCOL)



        architecture_file_path = Embedding_Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size # can't be < than batch size

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
            idx = 1  # default [UNK]
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
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sample_token_idx

            states_value = [h, c]
        return target_text.strip()

    def n_max(self, arr, n):
        """Argmax for n most common values returns indxs"""
        return list(arr.argsort()[-n:][::-1])


    def choose_n_hypos(self, tmp_hypos, n):
        hypo_weights = []
        for hypo in tmp_hypos:
            hypo_weight = 0
            for w in hypo:
                hypo_weight+=w[1]
            hypo_weights.append(hypo_weight)
        idxs = self.n_max(np.array(hypo_weights), n)
        result = []
        for idx in idxs:
            result.append(tmp_hypos[idx])
        return result


    def decode_beam(self, hypo):
        """Convert hypotheses to natural language"""
        decoded_sentence = ''
        for h in hypo:
            word = self.target_idx2word[h[0]]
            decoded_sentence += ' ' + word
        # return decoded_sentence
        return re.sub(' END', '', decoded_sentence)


    def beam_step(self, hypotheses, next_states_value, beam_size):
        """We have BEAM (3) hypotheses: [[(idx1, log_prob1)], [(idx2, log_prob2)], [(idx3, log_prob3)]]
           Now we want to extend them to BEAM**BEAM (from 3 to 9)"""
    
        tmp_hypos = []
        for hypo in hypotheses:
            # We predict based on last word in current hypo
            last_idx = hypo[-1][0] 
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = last_idx
            
            next_output_tokens, next_h, next_c = self.decoder_model.predict([target_seq] + next_states_value)
            next_sampled_token_indexes = self.n_max(next_output_tokens[0, -1, :], beam_size)
            for idx in next_sampled_token_indexes:
                log_prob = np.log(next_output_tokens[0, -1, idx])
                new_hypo = hypo + [(idx, log_prob)]
                tmp_hypos.append(new_hypo)
            
            # Update states
            next_states_value = [next_h, next_c]
    
        next_hypotheses = self.choose_n_hypos(tmp_hypos, beam_size)
    
        stop_condition = False

        # check if end of sequence achieved
        for hypo in next_hypotheses:
            if hypo[-1][0] == self.target_word2idx['END']:
                stop_condition = True 
    
        # if maxlen or end is acciebed -> finish
        if len(next_hypotheses[0]) == self.max_target_seq_length or stop_condition:
            return self.choose_n_hypos(next_hypotheses, 1)[0]
    
        else:
            return self.beam_step(next_hypotheses, next_states_value, beam_size)


    def beam_search(self, input_text, beam_size = 3):
        """Wrap up all beam search functions"""

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
        target_seq = np.zeros((1, 1)) # this is from https://github.com/devm2024/nmt_keras/blob/master/base.ipynb
        target_seq[0, 0] = self.target_word2idx['START']

        # predict (probability distriburion + new states values)
        output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

        # Sample n tokens
        sampled_token_indexes = self.n_max(output_tokens[0, -1, :], beam_size)
    
        # add their idxs and log_probs to hypotheses
        hypotheses = []
        for idx in sampled_token_indexes:
            log_prob = np.log(output_tokens[0, -1, idx])
            hypotheses.append([(idx, log_prob)])
        
        # AFTER THAT OUR HYPOTESES SHOULD LOOK LIKE THIS:
        # [[(idx1, log_prob1)], [(idx2, log_prob2)], [(idx3, log_prob3)]]
        # where each hypotheses is a list of tuples
        # each tuple is a word idx and its log_prob
      
        # Update states
        next_states_value = [h, c] 
    
        return self.decode_beam(self.beam_step(hypotheses, next_states_value, beam_size))

