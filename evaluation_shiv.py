import numpy as np
import torch
import dataset_shiv
from metric_shiv import *
import datetime
import random
import sys
import copy
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20, warm_start=5):
        self.model = model
        self.loss_func = loss_func

        self.warm_start = warm_start
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def evalRNN(self, eval_data, batch_size):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        # losses = None
        # recalls = None
        # mrrs = None

        dataloader = eval_data

        eval_iter = 0

        with torch.no_grad():
            total_test_num = []
            for input_x_batch, target_y_batch, x_len_batch, idx_batch in dataloader:
                input_x_batch = input_x_batch.to(self.device)
                target_y_batch = target_y_batch.to(self.device)
                warm_start_mask = (idx_batch>=self.warm_start)
                
                hidden = self.model.init_hidden()

                logit_batch, hidden = self.model(input_x_batch, hidden, x_len_batch)

                logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
                loss_batch = self.loss_func(logit_sampled_batch)

                losses.append(loss_batch.item())
                
                recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, warm_start_mask, k=self.topk)

                weights.append( int( warm_start_mask.int().sum() ) )
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)

                total_test_num.append(target_y_batch.view(-1).size(0))

        mean_losses = np.mean(losses)
        mean_recall = np.average(recalls, weights=weights)
        mean_mrr = np.average(mrrs, weights=weights)
        print("total_test_num", np.sum(total_test_num))

        return mean_losses, mean_recall, mean_mrr

    # def eval(self, eval_data, batch_size, debug=False):
    #     self.model.eval()

    #     losses = []
    #     recalls = []
    #     mrrs = []
    #     weights = []

    #     dataloader = eval_data

    #     eval_iter = 0

    #     with torch.no_grad():
    #         total_test_num = []
    #         for input_x_batch, target_y_batch, idx_batch in dataloader:
    #             input_x_batch = input_x_batch.to(self.device)
    #             target_y_batch = target_y_batch.to(self.device)
    #             warm_mask = (idx_batch >= self.warm_start)

    #             logit_batch = self.model(input_x_batch)

    #             logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
    #             loss_batch = self.loss_func(logit_sampled_batch)

    #             losses.append(loss_batch.item())

    #             recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, warm_mask, k=self.topk)

    #             weights.append(int(warm_mask.int().sum()))
    #             recalls.append(recall_batch)
    #             mrrs.append(mrr_batch)

    #              #flattens to 1D to then get total number of elements in target_y_batch
    #             total_test_num.append(target_y_batch.view(-1).size(0))

    #     mean_loss = np.mean(losses)
    #     mean_recall = np.mean(recalls)
    #     mean_mrr = np.mean(mrrs)

    #     return mean_loss, mean_recall, mean_mrr

    def eval(self, eval_data, batch_size, debug=False):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []
        
        item_popularity = defaultdict(int)
        item_losses = defaultdict(float)
        item_recalls = defaultdict(list)
        item_mrrs = defaultdict(list)

        dataloader = eval_data

        with torch.no_grad():
            total_test_num = []
            for input_x_batch, target_y_batch, idx_batch in dataloader:
                input_x_batch = input_x_batch.to(self.device) # batch size x sequence length
                target_y_batch = target_y_batch.to(self.device) # batch size
                warm_mask = (idx_batch >= self.warm_start) # batch size

                # Find popularity based on number of interactions in data
                for seq in input_x_batch:
                    for item in seq:
                        item_popularity[item.item()] += 1
                for item in target_y_batch:
                    item_popularity[item.item()] += 1

                logit_batch = self.model(input_x_batch) # batch size x number of items

                logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)] # batch size x batch size
                    
                loss_batch = self.loss_func(logit_sampled_batch)
                losses.append(loss_batch.item())

                recall_batch, item_rec, mrr_batch, item_mrr = evaluate(logit_batch, target_y_batch, warm_mask, k=self.topk)
                
                for k, v in item_rec.items():
                    item_recalls[k].append(v)

                for k, v in item_mrr.items():
                    item_mrrs[k].append(v)
                
                weights.append(int(warm_mask.int().sum()))
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)
                
                #flattens to 1D to then get total number of elements in target_y_batch
                total_test_num.append(target_y_batch.view(-1).size(0))

        for k in item_popularity.keys():
            if k not in item_recalls:
                item_recalls[k].append(0)
            if k not in item_mrrs:
                item_mrrs[k].append(0)

        for k, v in item_recalls.items():
            item_recalls[k] = np.mean(v)

        for k, v in item_mrrs.items():
            item_mrrs[k] = np.mean(v)

        recall_popularities = defaultdict(list)
        mrr_popularities = defaultdict(list)

        for k, v in item_popularity.items():
            if v < 100:
                recall_popularities[v].append(item_recalls[k])
                mrr_popularities[v].append(item_mrrs[k])

        for v in recall_popularities.keys():
            recall_vals = recall_popularities[v]
            mrr_vals = mrr_popularities[v]
            recall_popularities[v] = np.mean(recall_vals)
            mrr_popularities[v] = np.mean(mrr_vals)

        recall_fig, ax = plt.subplots()
        ax.bar(list(recall_popularities.keys()), list(recall_popularities.values()))
        ax.set(xlabel='popularity', ylabel='Recall@20',
            title='Recall of items with varying popularity')
        ax.grid()
        recall_fig.savefig("recall_popularity.png")
        plt.show()

        mrr_fig, axm = plt.subplots()
        axm.bar(list(mrr_popularities.keys()), list(mrr_popularities.values()))
        axm.set(xlabel='popularity', ylabel='MRR@20',
            title='MRR of items with varying popularity')
        axm.grid()
        mrr_fig.savefig("mrr_popularity.png")
        plt.show()

        mean_loss = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        return mean_loss, mean_recall, mean_mrr

    def eval_pred(self, model, dataset, args):
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        if usernum>10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)
        for u in users:

            if len(train[u]) < 1 or len(test[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

        return NDCG / valid_user, HT / valid_user