
from pyparsing import AtStringStart
import torch
import numpy as np
import torch.nn as nn
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import random
import math
from model import GPT,Transformer

class Corpus(object):
    def __init__(self, path, batch_size, max_sql):
        self.vocabulary = []
        self.word_id = {}
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.dset_flag = "train"

        ## max_sql means the maximum sequence length
        self.max_sql = max_sql
        self.batch_size = batch_size
        print("size of train set: ", self.train.size(0))
        print("size of valid set: ", self.valid.size(0))
        self.train_batch_num = self.train.size(0) // self.batch_size["train"]
        self.valid_batch_num = self.valid.size(0) // self.batch_size["valid"]
        self.train = self.train.narrow(
            0, 0, self.batch_size["train"] * self.train_batch_num)
        self.valid = self.valid.narrow(
            0, 0, self.batch_size["valid"] * self.valid_batch_num)
        self.train = self.train.view(self.batch_size["train"], -1).contiguous()
        self.valid = self.valid.view(self.batch_size["valid"], -1).contiguous()

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    @property
    def pad_idx(self):
        if '<pad>' not in self.word_id:
            return None
        else:
            return self.word_id['<pad>']

    def set_train(self):
        self.dset_flag = "train"
        self.train_si = 0

    def set_valid(self):
        self.dset_flag = "valid"
        self.valid_si = 0

    def tokenize(self, file_name):
        file_lines = open(file_name, 'r').readlines()
        num_of_words = 0
        for line in file_lines:
            words = ['<bos>'] + line.split() + ['<eos>']
            num_of_words += len(words)
            for word in words:
                if word not in self.word_id:
                    self.word_id[word] = len(self.vocabulary)
                    self.vocabulary.append(word)
        file_tokens = torch.LongTensor(num_of_words)
        token_id = 0
        for line in file_lines:
            words = ['<bos>'] + line.split() + ['<eos>']
            for word in words:
                file_tokens[token_id] = self.word_id[word]
                token_id += 1
        return file_tokens

    def get_batch(self):
        ## train_si and valid_si indicates the index of the start point of the current mini-batch
        if self.dset_flag == "train":
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(1)-self.train_si-1)
            data_loader = self.train
            self.train_si = self.train_si + seq_len
        else:
            start_index = self.valid_si
            seq_len = min(self.max_sql, self.valid.size(1)-self.valid_si-1)
            data_loader = self.valid
            self.valid_si = self.valid_si + seq_len
        data = data_loader[:, start_index:start_index+seq_len]
        target = data_loader[:, start_index+1:start_index+seq_len+1]

        ## end_flag indicates whether a epoch (train or valid epoch) has been ended
        if self.dset_flag == "train" and self.train_si+1 == self.train.size(1):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == "valid" and self.valid_si+1 == self.valid.size(1):
            end_flag = True
            self.valid_si = 0
        else:
            end_flag = False
        return data, target, end_flag



if __name__ == '__main__':
    device = torch.device('cpu')
    parser = argparse.ArgumentParser(
        description='Test PyTorch ptb Language Model with Transformer')
    parser.add_argument('--d_model', default=512, type=int,
                        help='the dimension of the word embeddings')
    parser.add_argument('--n_heads', default=8, type=int,
                        help='the number of heads')
    parser.add_argument('--n_layers', default=6, type=int,
                        help='the number of the encoder & decoder layers')
    parser.add_argument('--max_sql', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='dropout probability')
    parser.add_argument('--d_hidden', default=2048, type=int,
                        help='FeedForward Dimension')
    parser.add_argument('--model', type=str, choices=[
                        'transformer', 'GPT'], default='GPT', help="choose which model to use")
    args = parser.parse_args()
    
    train_batch_size = 1
    eval_batch_size = 1
    batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
    # data_loader = Corpus("../data/ptb", batch_size, args.max_sql)
    data_loader = Corpus('../data/ptb', batch_size, args.max_sql)
    n_heads = args.n_heads
    src_vocab_size = data_loader.vocab_size
    target_vocab_size = src_vocab_size
    d_model = args.d_model
    n_layers = args.n_layers
    d_hidden = args.d_hidden
    dropout = args.dropout
    if args.model == 'transformer':
        model = Transformer(src_vocab_size, target_vocab_size, d_model, d_hidden, n_heads,
                            n_layers, dropout, args.max_sql+10, data_loader.pad_idx, data_loader.pad_idx,device)
    else:
        model = GPT(src_vocab_size, args.max_sql, d_model, d_hidden,
                    n_heads, n_layers, dropout, data_loader.pad_idx,device)
    # print(model.state_dict())
    model.load_state_dict(torch.load(os.path.join('checkpoint',args.model+'-'+str(args.d_model)+'-'+str(args.n_heads)+'-'+str(args.d_hidden)+'-'+str(args.dropout)+'.pth')))
    model.to(device)
    model.eval()
    
    data_loader.set_train()

    for i in range(random.randint(1,10000)):
        data,target,end_flag = data_loader.get_batch()
    
    result,_,attn,_ = model(data,target)
    _,result = torch.max(result,-1)
    data = data.squeeze().numpy()
    target = target.squeeze().numpy()
    # print(attn[0])
    attn = attn[-1].squeeze().mean(dim=0).detach().numpy()
    result = result.numpy()
    # print(data.shape)
    # print(target.shape)
    # print(result.shape)
    # print(attn.shape)
    # print(attn)
    variables = [data_loader.vocabulary[id] for id in data]
    labels = [data_loader.vocabulary[id] for id in result]
    # print(labels)
    # labels = ['ID_0','ID_1','ID_2','ID_3']
    # attn = np.tril(attn,k=0)
    # variables = variables[:10]
    # labels = labels[:10]
    # print(type(attn))
    # print(attn)
    # attn = attn[:10,:10]
    df = pd.DataFrame(attn, columns=variables, index=variables)
    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='binary')
    fig.colorbar(cax)

    tick_spacing = 1
    # plt.tick_params(labelsize=30)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([''] + list(df.columns),rotation=90)
    ax.set_yticklabels([''] + list(df.index))

    # plt.show()
    plt.savefig(os.path.join('figures','attention_map.png'),bbox_inches='tight')
    