from comet_ml import Experiment
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import math
import numpy as np
from preprocess import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
     "batch_size": 50,
     "num_epochs": 5,
     "learning_rate": 0.0001,
     "window_size": 40,
 }

def train(model, train_loader, optimizer, loss_fn, experiment):
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
        for e in range(hyper_params['num_epochs']):
            correct = 0
            total = 0
            total_loss = 0
            for item in tqdm(train_loader):
                    x = item['input']
                    y = item['label']
                    seq = x.to(device)
                    if len(seq[0]) == 1:
                        continue
                    labels = y.to(device)
                    mask = item['mask']
                    mask = mask.to(device)
                    optimizer.zero_grad()

                    loss, output = model(seq, attention_mask=mask, labels=seq)[:2]
                    loss = loss.to(device)
                    output = output.to(device)
                    labels = labels.view(-1)
                    output = output.view(-1, output.shape[2])
                    #loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(output.data, 1)

                    #print(predicted) 
                    total += labels.size(0)
                    total_loss += loss.item()*labels.size()[0]
                    correct += (predicted == labels.data).sum().float()

            print('epoch:', e+1)
            print('accuracy', correct/total)
            print('loss:', loss)
            print('perplexity:', total_loss/total)


def test(model, test_loader, experiment):
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
        word_count = 0
        correct = 0

        with torch.no_grad():
            model = model.eval()
            with experiment.test():
                for item in tqdm(test_loader):
                    x = item['input']
                    seq = x.to(device)
                    if len(seq[0]) == 1:
                        continue
                    y = item['label']
                    mask = item['mask']
                    labels = y.to(device)
                    mask = mask.to(device)

                    loss = model(seq, labels=seq, attention_mask=mask)[0]
                    labels = labels.view(-1)
                    total_loss += loss.item()*labels.size()[0]
                    word_count += labels.size()[0]

                    # _, predicted = torch.max(y_pred.data, 1)
                    # correct += (predicted == labels.data).sum().float()
                    torch.cuda.empty_cache()

            perplexity = math.exp(total_loss/word_count)
            # accuracy = (correct/word_count).item()

            print("perplexity:", perplexity)
            experiment.log_metric("perplexity", perplexity)

def determine_sentiment(scores):
    compound_score = scores['compound']
    if compound_score < -.2:
        return '<neg> '
    elif compound_score < .2:
        return '<neu> '
    else:
        return '<pos> '


def interactive(input, tokenizer, model, top_k=8, ntok=20, mode='neutral'):
    """
    Generate and print out the response given input using the trained model
    :param input: an input string as prompt (i.e. How are you?)
    :param tokenizer: intialized tokenizer object for encoding the input
    :param model: the trained model to use for generate prediction
    :param top_k: number of samples for top_l sampling
    :param ntok: maximum number of tokens to generate

    Comment: Feed in the input to the model to generate the most probable
    token and concatenate it with current input.
    Continue this process iteratively until the model predicts the padding
    token or reach the maximum number of tokens.
    You may need to add the BOS token and special token to the input sentence
    before passing into model.
    Also, you may want to filter out your input sentence and meaningless tokens
    when printing out the response.
    """
    # TODO: Write the generation function for interacting with trained model

    modes = {'neutral': '<neu> ', 'positive': '<pos> ', 'negative': '<neg> '}
    if mode not in modes:
        r_sent = modes['neutral']
    else:
        r_sent = modes[mode]

    prompt = input
    sid = SentimentIntensityAnalyzer()
    p_scores = sid.polarity_scores(prompt)
    p_sent = determine_sentiment(p_scores)

    prompt = p_sent + prompt + ' <s> ' + r_sent
    enc = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)).cuda()
    response = []

    while len(response) < ntok:
        with torch.no_grad():
            logits = model(enc)[0]
            #logits = logits.view(-1, logits.shape[1])
            topk, inds = torch.topk(logits, k=top_k)
            new_logits = Variable(torch.zeros(logits.size(0), logits.size(1))).cuda()
            logits = new_logits.scatter(1, inds, topk)

            probs = F.relu(logits[-1])
            predicted = torch.multinomial(probs, 1)

            #new_word = predicted[-1].unsqueeze(0)
            if predicted.item() == tokenizer.pad_token_id:
                break
            else:
                enc = torch.cat((enc, predicted), 0)
                response.append(predicted.item())

    decoded = tokenizer.decode(response, skip_special_tokens=True)
    print("Response:", decoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("mode", default='neutral')
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

    experiment = Experiment(log_code=False,
         api_key='6f91f3AzPkFoIDvp9njF2QwET',
         workspace='rkty1obt',
         project_name='emotional_chatbot',
         display_summary=False)
    experiment.log_parameters(hyper_params)

    # Load the GPT2 Tokenizer, add any special token if needed

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens_dict = {'bos_token': '<b>', 'pad_token': '<p>', 'sep_token': '<s>', \
                    'additional_special_tokens': ['<neg>', '<neu>', '<pos>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Intialized the pretrained GPT-2 model and optimizer

    model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Load the train, test DataLoader NOTE: Parse the data using GPT2 tokenizer

    train_loader = load_dataset(args.train_file, tokenizer, hyper_params['batch_size'], hyper_params['window_size'])
    test_loader = load_dataset(args.test_file, tokenizer, hyper_params['batch_size'], hyper_params['window_size'])

    if args.train:
        # run train loop here
        print("running training loop...")
        train(model, train_loader, optimizer, loss_fn, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.test:
        # run test loop here
        print("running testing loop...")
        test(model, test_loader, experiment)
    if args.interactive:
        # generate your own chat with the model here
        print("running interative mode...")
        while True:
            input_text = input("Please say something: ")
            interactive(input_text, tokenizer, model, mode=args.mode)
