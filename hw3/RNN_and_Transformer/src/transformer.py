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
import os.path as osp
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
# device = torch.device('cpu')


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
########################################

# positional embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k, pad_idx: int = 0):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列(跟注意力机制的q,k没有关系)例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数,所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    # [batch_size, 1, len_k], True is masked
    # print(pad_idx)
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device)


def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask=None, eps=1e-12):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        e: eps
        说明: 在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        batch_size, n_heads, length, d_tensor = K.size()
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = (Q @ K.transpose(-1, -2)) / np.sqrt(d_tensor)
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -eps)
        scores.masked_fill_(attn_mask, -1e9)

        attn = self.softmax(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = attn @ V

        context = self.dropout(context)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self, d_model, n_heads, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.W_Q = nn.Linear(self.d_model, self.d_heads *
                             self.n_heads)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(self.d_model, self.d_heads * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_heads * self.n_heads)
        self.fc = nn.Linear(self.d_heads * self.n_heads, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        batch_size = input_Q.size(0)
        # residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        Q = self.W_Q(input_Q).view(batch_size,  -1,
                                   self.n_heads, self.d_heads).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size,  -1,
                                   self.n_heads, self.d_heads).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size,  -1,
                                   self.n_heads, self.d_heads).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_heads], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_heads] -> [batch_size, len_q, n_heads * d_heads]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.d_heads)
        # 再做一个projection
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.dropout(output)
        # return nn.LayerNorm(d_model).to(device)(output + residual), attn
        return output, attn


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        # residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        # return nn.LayerNorm(d_model).to(device)(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, dropout=0):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PoswiseFeedForwardNet(d_model, d_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        _x = enc_inputs
        x, attn = self.attn(enc_inputs, enc_inputs,
                            enc_inputs, attn_mask=enc_self_attn_mask)
        x = self.norm1(x+_x)
        x = self.dropout1(x)
        # enc_outputs: [batch_size, src_len, d_model]
        _x = x
        x = self.ffn(x)
        x = self.norm2(x+_x)
        x = self.dropout2(x)
        return x, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.enc_dec_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PoswiseFeedForwardNet(d_model, d_hidden, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        _x = dec_inputs
        x, dec_self_attn = self.self_attn(
            dec_inputs, dec_inputs, dec_inputs, attn_mask=dec_self_attn_mask)
        x = self.norm1(x+_x)
        x = self.dropout1(x)

        if enc_outputs is not None:
            _x = x
            x, dec_enc_attn = self.enc_dec_attention(
                x, enc_outputs, enc_outputs, dec_enc_attn_mask)
            x = self.norm2(x+_x)
            x = self.dropout2(x)
        else:
            dec_enc_attn = None
        _x = x
        x = self.ffn(x)
        x = self.norm3(x+_x)
        x = self.dropout3(x)

        # dec_self_attn, dec_enc_attn这两个是为了可视化的
        return x, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, n_voc, max_len, d_model, d_hidden, n_heads, n_layers, dropout, pad_idx=0):
        super(Encoder, self).__init__()
        self.pad_idx = pad_idx
        # print(self.pad_idx)
        self.src_emb = nn.Embedding(n_voc, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(
            enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(
            enc_inputs, enc_inputs, self.pad_idx)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class GPT(nn.Module):
    def __init__(self, n_voc, max_len, d_model, d_hidden, n_heads, n_layers, dropout, pad_idx):
        super(GPT, self).__init__()
        if pad_idx is None:
            pad_idx = -1
        self.seq_emb = nn.Embedding(n_voc, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)
        self.n_voc = n_voc
        self.max_len = max_len
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.pad_idx = pad_idx
        self.layers = nn.ModuleList([DecoderLayer(
            d_model, d_hidden, n_heads, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, n_voc)

    def forward(self, inputs, _):
        outputs = self.seq_emb(inputs)
        outputs = self.pos_emb(outputs.transpose(0, 1)).transpose(0, 1)
        # [batch_size, target_len, target_len]
        self_attn_pad_mask = get_attn_pad_mask(inputs, inputs, self.pad_idx)
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        self_attn_subsequence_mask = get_attn_subsequence_mask(
            inputs)  # [batch_size, target_len, target_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        # [batch_size, target_len, target_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0
        self_attn_mask = torch.gt(
            (self_attn_pad_mask + self_attn_subsequence_mask), 0)

        self_attns = []
        for layer in self.layers:
            outputs, self_attn, _ = layer(outputs, None, self_attn_mask, None)
            self_attns.append(self_attn)
        # dec_outputs: [batch_size, target_len, d_model]
        outputs = self.fc(outputs)
        return outputs.view(-1, self.n_voc), None, self_attns, None


class Decoder(nn.Module):
    def __init__(self, n_voc, max_len, d_model, d_hidden, n_heads, n_layers, dropout, src_pad_idx=0, target_pad_idx=0):
        super(Decoder, self).__init__()
        self.target_emb = nn.Embedding(n_voc, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.layers = nn.ModuleList([DecoderLayer(
            d_model, d_hidden, n_heads, dropout) for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, target_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.target_emb(
            dec_inputs)  # [batch_size, target_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(
            0, 1)  # [batch_size, target_len, d_model]
        # Decoder输入序列的pad mask矩阵
        dec_self_attn_pad_mask = get_attn_pad_mask(
            dec_inputs, dec_inputs, self.target_pad_idx)  # [batch_size, target_len, target_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs)  # [batch_size, target_len, target_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        # [batch_size, target_len, target_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs, self.src_pad_idx)  # [batc_size, target_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, target_len, d_model], dec_self_attn: [batch_size, n_heads, target_len, target_len], dec_enc_attn: [batch_size, h_heads, target_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, target_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, n_srcvoc, n_targetvoc, d_model, d_hidden, n_heads, n_layers, dropout=0.1, max_len=50, src_pad_idx=0, target_pad_idx=0):
        super(Transformer, self).__init__()
        if src_pad_idx is None:
            src_pad_idx = -1
        if target_pad_idx is None:
            target_pad_idx = -1
        self.encoder = Encoder(n_srcvoc, max_len, d_model,
                               d_hidden, n_heads, n_layers, dropout, src_pad_idx)
        self.decoder = Decoder(n_targetvoc, max_len, d_model, d_hidden,
                               n_heads, n_layers, dropout, src_pad_idx, target_pad_idx)
        self.projection = nn.Linear(d_model, n_targetvoc)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入:两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


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
    parser.add_argument('--learning_rate', default=1e-3,
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
                            n_layers, dropout, args.max_sql+10, data_loader.pad_idx, data_loader.pad_idx)
    else:
        model = GPT(src_vocab_size, args.max_sql, d_model, d_hidden,
                    n_heads, n_layers, dropout, data_loader.pad_idx)
    model = model.to(device)
    logger, writer = setup_default_logging(args, args.model+'LM')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    # Loop over epochs.
    for epoch in range(1, args.epochs+1):
        # train()
        train(epoch, model, data_loader, criterion,
              optimizer, scheduler, logger, writer, device)
        evaluate(epoch, model, data_loader, criterion, logger, writer, device)
        # evaluate()
    writer.close()
