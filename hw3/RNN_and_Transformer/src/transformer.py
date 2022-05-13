from cmath import inf
import torch
import torch.nn as nn
import math
import numpy as np
import argparse
# Transformer模型
from logger import setup_default_logging
import os
import torch
from logging import Logger
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
# device = torch.device('cpu')
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


# Train Function
def train(num_epoch: int, model: nn.Module, data_loader: Corpus, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, logger: Logger, writer: SummaryWriter, device=torch.device('cpu')):
    costs = 0.0
    iters = 0
    model.train()
    data_loader.set_train()
    num_iter = 0
    while True:
        num_iter += 1
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        # print(data.shape)
        #############################################
        output, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            data, target)
        #############################################
        loss = criterion(
            output.view(-1, data_loader.vocab_size), target.reshape(-1))
        costs += loss.item() * data_loader.max_sql
        iters += data_loader.max_sql
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if num_iter % (data_loader.train_batch_num//(5*data_loader.max_sql)) == 0:
            logger.info("Epoch:[{}/{}]:{:.0%}, Loss:{:8.2f}, Perplexity: {:8.2f}".format(num_epoch, args.epochs,
                        data_loader.max_sql * num_iter * 1.0 / data_loader.train_batch_num, loss.item(), np.exp(loss.item())))
        if end_flag == True:
            break
    perplexity = np.exp(costs / iters)
    logger.info('Train perplexity at epoch {}: {:8.2f}'.format(
        num_epoch, perplexity))
    writer.add_scalar('train/1.loss', costs/iters, num_epoch)
    writer.add_scalar('train/2.perplexity', perplexity, num_epoch)
    if scheduler is not None:
        scheduler.step()
    return perplexity

#######################################
# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.


def evaluate(num_epoch: int, model: nn.Module, data_loader: Corpus, criterion: nn.Module, logger: Logger, writer: SummaryWriter, device=torch.device('cpu')):
    costs = 0.0
    iters = 0
    model.eval()
    data_loader.set_valid()
    with torch.no_grad():
        num_iter = 0
        while True:
            num_iter += 1
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            #############################################
            output, enc_self_attns, dec_self_attns, dec_enc_attns = model(
                data, target)
            # print(dec_self_attns[0])
            #############################################
            loss = criterion(
                output.view(-1, data_loader.vocab_size), target.reshape(-1))
            costs += loss.item() * data_loader.max_sql
            iters += data_loader.max_sql
            # log
            # if num_iter%(data_loader.valid_batch_num//(10*data_loader.max_sql))==0:
            #     logger.info("Epoch:[{}/{}]:{:.0%}, Loss:{:8.2f}, Perplexity: {:8.2f}".format(num_epoch,args.epochs,data_loader.max_sql*num_iter * 1.0 / data_loader.valid_batch_num, loss.item(),np.exp(loss.item())))
            if end_flag == True:
                break
    perplexity = np.exp(costs / iters)
    logger.info('Valid perplexity at epoch {}: {:8.2f}'.format(
        num_epoch, perplexity))
    writer.add_scalar('valid/1.loss', costs/iters, num_epoch)
    writer.add_scalar('valid/2.perplexity', perplexity, num_epoch)
    return perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch ptb Language Model with Transformer')
    parser.add_argument('--epochs', type=int, default=128,
                        help='upper epoch limit')
    parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                        help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                        help='evaluate batch size')
    parser.add_argument('--max_sql', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--seed', type=int, default=1234,
                        help='set random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA device')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU device id used')
    parser.add_argument('--learning_rate', default=1e-4,
                        type=float, help='learning rate')
    parser.add_argument('--d_model', default=512, type=int,
                        help='the dimension of the word embeddings')
    parser.add_argument('--n_heads', default=8, type=int,
                        help='the number of heads')
    parser.add_argument('--n_layers', default=6, type=int,
                        help='the number of the encoder & decoder layers')
    parser.add_argument('--dropout', default=0, type=float,
                        help='dropout probability')
    parser.add_argument('--d_hidden', default=2048, type=int,
                        help='FeedForward Dimension')
    parser.add_argument('--model', type=str, choices=[
                        'transformer', 'GPT'], default='GPT', help="choose which model to use")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    # Use gpu or cpu to train
    use_gpu = args.cuda
    global device
    if use_gpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(args.gpu_id)
    else:
        device = torch.device("cpu")

    # load data
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
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
    if data_loader.pad_idx is not None:
        criterion = nn.CrossEntropyLoss(ignore_index=data_loader.pad_idx)
    else:
        criterion = nn.CrossEntropyLoss()
    if args.model == 'transformer':
        model = Transformer(src_vocab_size, target_vocab_size, d_model, d_hidden, n_heads,
                            n_layers, dropout, args.max_sql+10, data_loader.pad_idx, data_loader.pad_idx,device)
    else:
        model = GPT(src_vocab_size, args.max_sql, d_model, d_hidden,
                    n_heads, n_layers, dropout, data_loader.pad_idx,device)
    model = model.to(device)
    logger, writer = setup_default_logging(args, args.model+'LM')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
    min_ppl = inf
    # Loop over epochs.
    for epoch in range(1, args.epochs+1):
        # train()

        train(epoch, model, data_loader, criterion,
              optimizer, scheduler, logger, writer, device)
        ppl = evaluate(epoch, model, data_loader, criterion, logger, writer, device)
        if ppl < min_ppl:
            min_ppl = ppl
            torch.save(model.state_dict(),os.path.join('checkpoint',args.model+'-'+str(args.d_model)+'-'+str(args.n_heads)+'-'+str(args.d_hidden)+'-'+str(args.dropout)+'.pth'))
        # evaluate()
    writer.close()
