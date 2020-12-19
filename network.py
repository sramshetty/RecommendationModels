from torch import nn
import torch
import torch.nn.functional as F
import datetime
import numpy as np
from model import PositionalEncoder, FeedForward, Transformer 

class GRU4REC(nn.Module):
    def __init__(self, window_size, input_size, hidden_size, output_size, num_layers=1, final_act='tanh', dropout_hidden=.8, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False, shared_embedding=True):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.window_size = window_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        
        self.h2o = nn.Linear(hidden_size, output_size)
            
        self.create_final_activation(final_act)

        self.look_up = nn.Embedding(input_size, self.embedding_dim).to(self.device)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)




        if shared_embedding:
            print("share embedding")
            self.out_matrix = self.look_up.weight.to(self.device)



            










        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden, input_len):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

    
            # if len(input.size()) == 2:
            #     embedded = input.unsqueeze(0)
            # print("embedding size", embedded.size())
        embedded = input
        embedded = self.look_up(embedded)
    
        # embedded = embedded.transpose(0, 1)

        # print("embedded size", embedded.size(), input_len.shape)
        embedded_pad = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=True)
        # print("embedded_pad size", embedded_pad.size())
        output, hidden = self.gru(embedded_pad, hidden) # (sequence, B, H)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = output.contiguous()
        # print("output size", output.size())

        last_output = output[-1, :, :]
        # print("lastoutput size", last_output.size())

        last_output = last_output.view(-1, last_output.size(-1))  # (B,H)
        output = self.h2o(last_output)
#         logit = self.final_activation(output) ## (B, output_size)
        logit = output
        return logit, hidden


    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """

        self.onehot_buffer.zero_()

        index = input.unsqueeze(2)
        # index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(2, index, 1)

        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), input.size(1), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
        

class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=1, use_cuda=True, batch_size=50, dropout_input=0, dropout_hidden=0.5, embedding_dim=-1, position_embedding=False, shared_embedding=True):
        super().__init__()

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        print("Beginning")
        self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0).to(self.device)
        print("Finshed Embedding")
        self.pe = PositionalEncoder(hidden_size) if position_embedding else None
        print("Finished Encoding inputs")
        self.encode_layers = nn.ModuleList([Transformer(hidden_size, num_heads, dropout=dropout_hidden) for i in range(num_layers)])
        print("Encoded Layers")
        self.decode = Transformer(hidden_size, num_heads, dropout=dropout_hidden)
        print("Decoded")

        if shared_embedding:
            self.out_matrix = self.embed.weight.to(self.device)
        else:
            self.out_matrix = torch.rand(hidden_size, output_size, requires_grad=True).to(self.device)

        self = self.to(self.device)

    def forward(self, src):
        x = self.embed(src)
        src_mask = (src == 0)
        if self.pe != None:
            x = self.pe(x)

        x = x.transpose(0,1)
        for i, layer in enumerate(self.encode_layers):
            x = layer(x, x, x, src_mask) ### encoded input sequence

        trg = self.embed(src[:, -1]).unsqueeze(0) ### last input
        d_output = self.decode(trg, x, x, src_mask)

        output = F.linear(d_output.squeeze(0), self.out_matrix)

        return output   

#My SASRec implementation
class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device
        
        self.item_emb = nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pe = nn.Embedding(args.maxlen, args.hidden_units)
        self.attn_blocks = nn.ModuleList([Transformer(args.hidden_units, args.num_heads, dropout=.2) for i in range(args.num_blocks)])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(args.hidden_units, eps=1e-8) for i in range(args.num_blocks)])
        self.dropout = nn.Dropout(args.dropout_rate)

        self.final_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        self = self.to(self.device)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pe(torch.LongTensor(positions).to(self.device))
        seqs = self.dropout(seqs)

        src_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~src_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i, block in enumerate(self.attn_blocks):
            seqs = seqs.transpose(0,1)
            q = self.attn_norms[i](seqs)
            seqs = block(q, seqs, seqs, attention_mask) ### encoded input sequence
            seqs *= ~src_mask.unsqueeze(-1)

        log_feats = self.final_norm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


 
 # https://github.com/pmixer/SASRec.pytorch/blob/master/model.py
class SASRecPM(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = FeedForward(args.hidden_units, dropout=args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)