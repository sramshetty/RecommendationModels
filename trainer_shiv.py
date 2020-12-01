from evaluation_shiv import *
import time
import torch
import numpy as np
import os
from dataset_shiv import *
import datetime

class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, topk, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.topk = topk
        self.args = args
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def train(self, start_epoch, end_epoch, batch_size, start_time=None):
        if start_time == None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch+1):
            print("*"*7,  epoch, "*"*7)
            start = time.time()
            train_loss = self.train_epoch(epoch, batch_size)

            loss, recall, mrr = self.evaluation_shiv.eval(self.train_data, batch_size)
            print("Train Epoch: {}, train loss: {:.4f},  loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - start))

            loss, recall, mrr = self.evaluation_shiv.eval(self.eval_data, batch_size, "Test")
            print("Test  Epoch: {}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, loss, recall, mrr, time.time() - start))

    def train_epoch(self, epoch, batch_size):
        self.model.train()
        losses = []
        torch.autograd.set_detect_anomaly(False)

        dataloader = self.train_data
        for input_x_batch, target_y_batch, idx_batch in dataloader:
            input_x_batch = input_x_batch.to(self.device)
            target_y_batch = target_y_batch.to(self.device)

            self.optim.zero_grad()
        
            logit_batch = self.model(input_x_batch)

            logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
            loss_batch = self.loss_func(logit_sampled_batch)
            
            losses.append(loss_batch.item())
            loss_batch.backward()

            self.optim.step()
        return np.mean(losses)