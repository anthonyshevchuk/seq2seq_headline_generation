#!/usr/bin/env python3
# coding: utf-8

from flask import Flask, jsonify
from flask import request
from flask import render_template
import pandas as pd
import numpy as np
import pickle
import re
from seq2seq.seq2seq_model import Seq2SeqSummarizer
import tensorflow as tf


model_dir_path = 'seq2seq/models'
model_path = Embedding_Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)
with open(model_path, 'rb') as data:
    config = pickle.load(data)
 
summarizer = Embedding_Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Embedding_Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))
graph = tf.get_default_graph()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/generate_headline', methods=['GET', 'POST'])
def seq2seq_summarize():
    global graph
    with graph.as_default():
        if not request.json or not 'article' in request.json:
            abort(400)
        article = request.json['article']
        headline = summarizer.summarize(article)
        return jsonify({'article': article, 'headline': headline}), 201


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # COMMAND: curl -i -H "Content-Type: application/json" -X POST -d '{"article":"Sample article"}' http://localhost:5000/generate_headline
