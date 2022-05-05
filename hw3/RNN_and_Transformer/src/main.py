# coding: utf-8
import argparse
from cgi import print_environ
from pickletools import optimize
import sched
import time
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import logger
import data
import model
import os
from logging import Logger
import os.path as osp
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=128,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='evaluate batch size')
parser.add_argument('--max_sql', type=int, default=30,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', default=0, type=int, help='GPU device id used')
parser.add_argument('--embedding_dim', default=128, type=int,
                    help='the dimension of the word embeddings')
parser.add_argument('--hidden_dim', default=1500, type=int,
                    help='the dimension of the hidden layer of GRU')
parser.add_argument('--hidden_layer_num', default=2, type=int,
                    help='the number of the hidden layers')
parser.add_argument('--learning_rate',default=4e-3,type=float,help='learning rate')
parser.add_argument('--dropout',default=0.3,type=float,help='dropout')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = args.cuda

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)
input_dim = args.embedding_dim
hidden_dim = args.hidden_dim
hidden_layer_num = args.hidden_layer_num
num_vocabulary = len(data_loader.vocabulary)

# detach the hidden state to avoid backpropagation
def detach(h):
    return h.detach()


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train(num_epoch:int,model:nn.Module,data_loader: data.Corpus,criterion:nn.Module,optimizer:torch.optim.Optimizer,scheduler:torch.optim.lr_scheduler._LRScheduler,logger:Logger,writer:SummaryWriter):
    costs = 0.0
    iters = 0   
    model.train()
    data_loader.set_train()
    hidden = torch.zeros(hidden_layer_num,train_batch_size,hidden_dim).to(device)
    num_iter = 0 
    while True:
        num_iter += 1
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target  = target.to(device)
        hidden = detach(hidden)
        #############################################
        output,hidden = model(data,hidden)
        #############################################
        loss = criterion(output.view(-1,len(data_loader.vocabulary)),target.reshape(-1))
        costs += loss.item() * data_loader.max_sql
        iters += data_loader.max_sql
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(),0.5)
        optimizer.step()

        # log
        if num_iter%(data_loader.train_batch_num//(5*data_loader.max_sql))==0:
            logger.info("Epoch:[{}/{}]:{:.0%}, Loss:{:8.2f}, Perplexity: {:8.2f}".format(num_epoch,args.epochs, data_loader.max_sql* num_iter * 1.0 / data_loader.train_batch_num, loss.item(),np.exp(loss.item())))
        if end_flag==True:
            break
    perplexity=np.exp(costs / iters)
    logger.info('Train perplexity at epoch {}: {:8.2f}'.format(num_epoch, perplexity))
    writer.add_scalar('train/1.loss', costs/iters, num_epoch)
    writer.add_scalar('train/2.perplexity', perplexity, num_epoch)
    if scheduler is not None:
        scheduler.step()
    return perplexity
    
########################################


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(num_epoch:int,model:nn.Module,data_loader: data.Corpus,criterion:nn.Module,logger:Logger,writer:SummaryWriter):
    costs = 0.0
    iters = 0
    model.eval()
    data_loader.set_valid()
    with torch.no_grad():
        num_iter=0
        hidden =  torch.zeros(hidden_layer_num,eval_batch_size,hidden_dim).to(device)
        while True:
            num_iter +=1
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target  = target.to(device)
            hidden = detach(hidden)
            #############################################
            output,hidden = model(data,hidden)
            #############################################
            loss = criterion(output.view(-1,len(data_loader.vocabulary)),target.reshape(-1))
            costs += loss.item() * data_loader.max_sql
            iters += data_loader.max_sql
            # log
            # if num_iter%(data_loader.valid_batch_num//(10*data_loader.max_sql))==0:
            #     logger.info("Epoch:[{}/{}]:{:.0%}, Loss:{:8.2f}, Perplexity: {:8.2f}".format(num_epoch,args.epochs,data_loader.max_sql*num_iter * 1.0 / data_loader.valid_batch_num, loss.item(),np.exp(loss.item())))
            if end_flag==True:
                break
    perplexity=np.exp(costs / iters)
    logger.info('Valid perplexity at epoch {}: {:8.2f}'.format(num_epoch, perplexity))
    writer.add_scalar('valid/1.loss', costs/iters, num_epoch)
    writer.add_scalar('valid/2.perplexity', perplexity, num_epoch)
    return perplexity
########################################


if __name__ == '__main__':

    # WRITE CODE HERE within two '#' bar
    ########################################
    # Build LMModel model (bulid your language model here)
    lm_model = model.LMModel(num_vocabulary,input_dim,hidden_dim,hidden_layer_num,dropout=args.dropout)
    lm_model.to(device)
    ########################################

    criterion = nn.CrossEntropyLoss()

    logger,writer = logger.setup_default_logging(args)
    optimizer = torch.optim.Adam(lm_model.parameters(),lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.9)
    # Loop over epochs.
    for epoch in range(1, args.epochs+1):
        # train()
        train(epoch,lm_model,data_loader,criterion,optimizer,scheduler,logger,writer)
        evaluate(epoch,lm_model,data_loader,criterion,logger,writer)
        # evaluate()
    writer.close()
