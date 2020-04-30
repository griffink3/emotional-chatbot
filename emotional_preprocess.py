import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch

def determine_sentiment(scores):
    compound_score = scores['compound']
    if compound_score < -.2:
        return ' <neg> '
    elif compound_score < .2:
        return ' <neu> '
    else:
        return ' <pos> '

class GPT2Dataset(Dataset):
    def __init__(self, input_file, tokenizer, window_size):
        self.lines = []
        self.inputs = []
        self.labels = []
        self.window_size = window_size
        self.tokenizer = tokenizer
        self.masks = []

        sid = SentimentIntensityAnalyzer()

        if input_file: 
            with open(input_file, 'r') as f: 
                raw = f.read().strip().split('\n')
                for line in raw: 
                    if line.find('persona') != -1:
                        pass
                    else:
                        line = line.strip().split('\t')
                        prompt = line[0][2:]
                        response = line[1]

                        p_scores = sid.polarity_scores(prompt)
                        prompt_sent = determine_sentiment(p_scores)
                        r_scores = sid.polarity_scores(response)
                        response_sent = determine_sentiment(r_scores)

                        new_line = prompt + ' <s> ' + response_sent + response

                        enc_dict = tokenizer.encode_plus(prompt_sent + new_line, \
                            max_length=window_size, pad_to_max_length=True, add_special_tokens=True, return_attention_mask=True)
                        x = enc_dict['input_ids']
                        mask = enc_dict['attention_mask']
                        y = tokenizer.encode(new_line + tokenizer.eos_token, \
                            max_length=window_size, pad_to_max_length=True, add_special_tokens=True)

                        self.inputs.append(torch.tensor(x))
                        self.labels.append(torch.tensor(y))
                        self.masks.append(torch.tensor(mask))


    def __len__(self):
        #return len(self.lines) // self.window_size
        return len(self.inputs)

    def __getitem__(self, idx):

        # sentence = self.lines[idx*self.window_size : (idx+1)*self.window_size]
        # x = self.tokenizer.encode(sentence, max_length=self.window_size, add_special_tokens=True)
        # y = self.tokenizer.encode(sentence, max_length=self.window_size, add_special_tokens=True)

        # item = {"input": torch.tensor(x), "label": torch.tensor(y)}

        item = {
        "input": self.inputs[idx],
        "label": self.labels[idx], 
        "mask": self.masks[idx]
        }

        return item 


def load_dataset(fn, tokenizer, batch_size, window_size):
    """
    :param fn: filename for the dataset
    :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for train and test
    :Comment: This function should be similary as the last GPT-2 assignemnent. We are still using the GPT-2 tokenizer to get the word ids.
    One thing to note is that when preprocessing the dataset, please exclude all the sentences defining the speaker's persona, in this assignment,
    we will just be training and testing on the chat text data. Also for each record of chat in the format of "sentence \t response", add BOS and EOS token
    to the start and end it, also add a special token between sentence and response to differentiate them from each other
    """

    data_set = GPT2Dataset(fn, tokenizer, window_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    return data_loader
