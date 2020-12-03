import numpy as np
import torch
import dataset_shiv
from metric_shiv import *
import datetime
import random
import sys
import copy

class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20, warm_start=5):
        self.model = model
        self.loss_func = loss_func

        self.warm_start = warm_start
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def evalRNN(self, eval_data, batch_size, train_test_flag):
		self.model.eval()

		losses = []
		recalls = []
		mrrs = []
		weights = []

		# losses = None
		# recalls = None
		# mrrs = None

		dataloader = eval_data

		with torch.no_grad():
			total_test_num = []

			for x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch, y_action_batch, y_action_idx_batch in dataloader:

				if train_test_flag == "train":
					eval_flag = random.randint(1,101)
					if eval_flag != 10:
						continue

				x_short_action_batch = x_short_action_batch.to(self.device)
				mask_short_action_batch = mask_short_action_batch.to(self.device)
				y_action_batch = y_action_batch.to(self.device)
			
				warm_start_mask = (y_action_idx_batch>=self.warm_start)
	
				output_batch = self.model(x_short_action_batch, mask_short_action_batch, pad_x_short_actionNum_batch)

				sampled_logit_batch, sampled_target_batch = self.model.m_ss(output_batch, y_action_batch, None, None, None, None, None, None, "full")

				loss_batch = self.loss_func(sampled_logit_batch, sampled_target_batch)
				losses.append(loss_batch.item())

				# et_2 = datetime.datetime.now()
				# print("duration 2", et_2-et_1)

				# logit_batch = self.model.m_ss.params(output_batch)
				recall_batch, mrr_batch = evaluate(sampled_logit_batch, sampled_target_batch, warm_start_mask, k=self.topk)

				weights.append( int( warm_start_mask.int().sum() ) )
				recalls.append(recall_batch)
				mrrs.append(mrr_batch)

				# et_3 = datetime.datetime.now()
				# print("duration 3", et_3-et_2)

				total_test_num.append(y_action_batch.view(-1).size(0))

		mean_losses = np.mean(losses)
		mean_recall = np.average(recalls, weights=weights)
		mean_mrr = np.average(mrrs, weights=weights)

		msg = "total_test_num"+str(np.sum(total_test_num))
		self.m_log.addOutput2IO(msg)

		return mean_losses, mean_recall, mean_mrr

    def eval(self, eval_data, batch_size, debug=False):
        self.model.eval()

        losses = []
        recalls = []
        mrrs = []
        weights = []

        dataloader = eval_data

        eval_iter = 0

        with torch.no_grad():
            total_test_num = []
            for input_x_batch, target_y_batch, idx_batch in dataloader:
                input_x_batch = input_x_batch.to(self.device)
                target_y_batch = target_y_batch.to(self.device)
                warm_mask = (idx_batch >= self.warm_start)

                logit_batch = self.model(input_x_batch)

                logit_sampled_batch = logit_batch[:, target_y_batch.view(-1)]
                loss_batch = self.loss_func(logit_sampled_batch)

                losses.append(loss_batch.item())

                recall_batch, mrr_batch = evaluate(logit_batch, target_y_batch, warm_mask, k=self.topk)

                weights.append(int(warm_mask.int().sum()))
                recalls.append(recall_batch)
                mrrs.append(mrr_batch)

                 #flattens to 1D to then get total number of elements in target_y_batch
                total_test_num.append(target_y_batch.view(-1).size(0))

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