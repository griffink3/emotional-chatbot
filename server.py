from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

import torch
import torch.nn as nn
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
tokenizer.add_special_tokens({'pad_token': '<p>', 'sep_token': '<s>'})
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('model.pt'))
model = model.eval()

def is_special_token(tokenizer, token_id):
    for special_id in tokenizer.all_special_ids:
        if special_id == token_id:
            return True
    return False

def processMessage(message):
    ntok = 15
    top_k = 5
    line = tokenizer.bos_token + ' ' + message + ' ' + tokenizer.sep_token
    tokenized_input = torch.tensor(tokenizer.encode(line, add_special_tokens=True, return_tensor=True)).to(device)
    response = ''
    with torch.no_grad():
        num_response_words = 0
        while num_response_words < ntok:
            output = model(tokenized_input)[0] # window_sz x vocab_sz
            _, vocab_indices = torch.topk(output[-1,:], top_k)
            perm = torch.randperm(top_k)
            perm_i = 0
            top_index = vocab_indices[perm[perm_i]]
            while is_special_token(tokenizer, top_index):
                if top_index == tokenizer.pad_token_id:
                    break
                perm_i += 1
                top_index = perm[perm_i]
            chosen_word = tokenizer.decode(torch.tensor([top_index]), skip_special_tokens=True)
            response += chosen_word + ' '
            tokenized_input = torch.cat((tokenized_input, torch.tensor([top_index]).to(device)))

            num_response_words += 1
    return response

@app.route("/")
def home():
    return "API for Cooper, the emotional chatbot."

@app.route("/send", methods=['GET'])
@cross_origin()
def process():
    message = request.args.get('text')
    if message == None:
        return 'ERROR'
    return processMessage(message)
    
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
