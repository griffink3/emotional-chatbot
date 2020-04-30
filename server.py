from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import math
import numpy as np
from preprocess import *
from tqdm import tqdm
from transformers import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 25,
     "num_epochs": 2,
     "learning_rate": 1E-4,
     "window_size": 20
}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens_dict = {'bos_token': '<b>', 'pad_token': '<p>', 'sep_token': '<s>', \
                    'additional_special_tokens': ['<neg>', '<neu>', '<pos>']}
tokenizer.add_special_tokens(special_tokens_dict)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('model.pt'))
model = model.eval()

def is_special_token(tokenizer, token_id):
    for special_id in tokenizer.all_special_ids:
        if special_id == token_id:
            return True
    return False

def processMessage(message, mode):
    ntok = 15
    top_k = 5

    modes = {'neutral': '<neu> ', 'positive': '<pos> ', 'negative': '<neg> '}
    if mode not in modes:
        r_sent = modes['positive']
    else:
        r_sent = modes[mode]

    sid = SentimentIntensityAnalyzer()
    p_scores = sid.polarity_scores(message)
    p_sent = determine_sentiment(p_scores)

    line = p_sent + message + ' <s> ' + r_sent
    tokenized_input = torch.tensor(tokenizer.encode(line, add_special_tokens=True)).to(device)
    response = []
    
    # with torch.no_grad():
    #     num_response_words = 0
    #     while num_response_words < ntok:
    #         output = model(tokenized_input)[0] # window_sz x vocab_sz
    #         _, vocab_indices = torch.topk(output[-1,:], top_k)
    #         perm = torch.randperm(top_k)
    #         perm_i = 0
    #         top_index = vocab_indices[perm[perm_i]]
    #         while is_special_token(tokenizer, top_index):
    #             if top_index == tokenizer.pad_token_id:
    #                 break
    #             perm_i += 1
    #             top_index = perm[perm_i]
    #         chosen_word = tokenizer.decode(torch.tensor([top_index]), skip_special_tokens=True)
    #         response += chosen_word + ' '
    #         tokenized_input = torch.cat((tokenized_input, torch.tensor([top_index]).to(device)))

    #         num_response_words += 1
    # return response

    while len(response) < ntok:
        with torch.no_grad():
            logits = model(tokenized_input)[0]
            topk, inds = torch.topk(logits, k=top_k)
            new_logits = Variable(torch.zeros(logits.size(0), logits.size(1))).cuda()
            logits = new_logits.scatter(1, inds, topk)

            probs = F.relu(logits[-1])
            predicted = torch.multinomial(probs, 1)
            if predicted.item() == tokenizer.pad_token_id:
                break
            else:
                tokenized_input = torch.cat((tokenized_input, predicted), 0)
                response.append(predicted.item())

    decoded = tokenizer.decode(response, skip_special_tokens=True)
    return decoded

def determine_sentiment(scores):
    compound_score = scores['compound']
    if compound_score < -.2:
        return '<neg> '
    elif compound_score < .2:
        return '<neu> '
    else:
        return '<pos> '

@app.route("/")
def home():
    return "API for Cooper, the emotional chatbot."

@app.route("/send", methods=['GET'])
@cross_origin()
def process():
    message = request.args.get('text')
    mode = request.args.get('mode')
    if message == None:
        return 'ERROR'
    if mode == None:
        mode = 'Positive'
    mode = mode.lower()
    return processMessage(message)
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
