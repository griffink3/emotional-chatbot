from comet_ml import Experiment
import torch
import torch.nn as nn
import argparse
import math
import numpy as np
from sentiment_preprocess import *
from tqdm import tqdm
from transformers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 25,
     "num_epochs": 2,
     "learning_rate": 1E-4,
     "window_size": 20
 }


def train(model, train_loader, optimizer, experiment, loss_fn):
    """
    Trains the model.
    :param model: the initilized model to use for forward and backward pass
    :param train_loader: Dataloader of training data
    :param optimizer: the initilized optimizer
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the training loop here, save trained model weights if needed
    model = model.train()    
    with experiment.train():
        for epoch in range(hyper_params['num_epochs']):
            num_correct = 0
            num_words = 0
            for i, items in enumerate(tqdm(train_loader)):
                x = items['x'].to(device)
                y = items['y'].to(device).view(-1)
                mask = items['mask'].to(device)

                optimizer.zero_grad()
                y_pred = model(x, attention_mask=mask)[0]
                y_pred = y_pred.view(-1, y_pred.shape[2])
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                num_words += y.size(0) 
                _, predictions = torch.max(y_pred.data, 1)
                num_correct += (predictions == y.data).sum()

                # Log accuracy/loss to Comet.ml using experiment.log_metric
                accuracy = (1.0 * num_correct / num_words).cpu().numpy()
                experiment.log_metric("accuracy", accuracy, step=i)
                experiment.log_metric("loss", loss.item(), step=i)
  


def test(model, test_loader, experiment, loss_fn):
    """
    Validates the model performance as LM on never-seen data using perplexity.
    :param model: the trained model to use for testing
    :param test_loader: Dataloader of testing data
    :param experiment: comet.ml experiment object
    """
    # TODO: Write the testing loop and calculate perplexity
    model = model.eval()
    with experiment.validate():
        total_loss = 0
        num_words = 0
        for i, items in enumerate(tqdm(test_loader)):
            x = items['x'].to(device)
            y = items['y'].to(device).view(-1)
            mask = items['mask'].to(device)

            y_pred = model(x, attention_mask=mask)[0]
            y_pred = y_pred.view(-1, y_pred.shape[2])
            loss = loss_fn(y_pred, y)

            num_words += y.size(0) 
            total_loss += loss.item()*y.size()[0]
        
        perplexity = math.exp(total_loss / num_words)
        print("perplexity: ", perplexity)
        experiment.log_metric("perplexity", perplexity) 

def is_special_token(tokenizer, token_id):
    for special_id in tokenizer.all_special_ids:
        if special_id == token_id:
            return True
    return False

def interactive(input, tokenizer, model, top_k=5, ntok=15):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_l sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable token
    and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model
    line = tokenizer.bos_token + ' ' + input + ' ' + tokenizer.sep_token
    tokenized_input = torch.tensor(tokenizer.encode(line, add_special_tokens=True, return_tensor=True)).to(device)
    response = ''
    model = model.eval()
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

        print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("train_file")
    # parser.add_argument("test_file")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="run in interactive mode")
    args = parser.parse_args()
    train_file = '../../data/train_both_revised_no_cands.txt'
    test_file = '../../data/valid_both_revised_no_cands.txt'

    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens_dict = {'pad_token': '<p>', 'sep_token': '<s>', 'additional_special_tokens': ['<neg>', '<neu>', '<pos>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load the GPT2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Intialized the pretrained GPT-2 model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate']) 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if args.load:
        model.load_state_dict(torch.load('model.pt'))
    if args.train:
        # run train loop here
        print("running training loop...")
        train_loader = DataLoader(ParsingDataset(train_file, tokenizer, hyper_params['window_size']), \
                                            batch_size=hyper_params['batch_size'], shuffle=False)
        train(model, train_loader, optimizer, experiment, loss_fn)
    if args.save:
        torch.save(model.state_dict(), 'model.pt')
    if args.test:
        # run test loop here
        print("running testing loop...")
        test_loader = DataLoader(ParsingDataset(test_file, tokenizer, hyper_params['window_size']), \
                                            batch_size=hyper_params['batch_size'], shuffle=False)
        test(model, test_loader, experiment, loss_fn)
    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, tokenizer, model)
