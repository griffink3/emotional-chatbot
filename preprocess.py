from transformers import *
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class ParsingDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_seq_len):
        """
        :return: (torch.utils.data.DataLoader, torch.utils.data.DataLoader) for 
        train and test
        :Comment: This preprocess step is different from the previous ones. 
        In this assignment, we are interested in using a pre-trained model.
        So, we have to use the exact vocabulary the pre-trained model was trained 
        with. We are using the GPT-2 model, so pass your data through
        the GPT-2 tokenizer to get the word ids. You don't need to create your 
        own dictionary.
        """
        self.dataset_length = 0
        self.X = []
        self.Y = []
        self.masks = []

        # TODO: read the input file line by line and put the lines in a list.
        file = open(input_file)
        lines = file.readlines()

        for line in lines:
            if line == '': continue
            split_line = line.strip().split('\t')
            if len(split_line) == 1: continue # no tab, not a prompt/response line

            self.dataset_length += 1
            start_i = split_line[0].find(' ')
            line = tokenizer.bos_token + split_line[0][start_i:] + ' ' + tokenizer.sep_token + ' ' + split_line[1]
            enc_dict = tokenizer.encode_plus(line, pad_to_max_length=True, add_special_tokens=True, max_length=max_seq_len, return_attention_mask=True)
            self.X.append(torch.tensor(enc_dict['input_ids']))
            self.masks.append(torch.tensor(enc_dict['attention_mask']))
            # dec_dict = tokenizer.encode_plus(line + ' ' + tokenizer.eos_token, pad_to_max_length=True, add_special_tokens=True, max_length=max_seq_len, return_attention_mask=False)
            self.Y.append(torch.tensor(enc_dict['input_ids'][1:] + [tokenizer.eos_token_id]))
            

    def __len__(self):
        """
        len should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return self.dataset_length

    def __getitem__(self, idx):
        """
        getitem should return a tuple or dictionary of the data at some index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the items in dataset
        return {
            'x': self.X[idx],
            'y': self.Y[idx],
            'mask': self.masks[idx]
        } 