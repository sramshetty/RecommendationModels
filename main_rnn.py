"""
command 
python main.py --data_folder ../Data/xing/ --train_data train_item.pickle --valid_data test_item.pickle --test_data test_item.pickle --data_name xing --embedding_dim 300 --hidden_size 300 --lr 0.005
"""

import argparse
import torch
# import lib
import numpy as np
import os
import datetime
from dataset_shiv import *
from loss_shiv import *
from model import *
from optimizer import *
from trainer_shiv import *
from network import GRU4REC
from torch.utils import data
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--dropout_input', default=0, type=float)
parser.add_argument('--dropout_hidden', default=.2, type=float)

# parse the optimizer arguments
parser.add_argument('--optimizer_type', default='Adagrad', type=str)
parser.add_argument('--final_act', default='tanh', type=str)
parser.add_argument('--lr', default=.05, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--eps', default=1e-6, type=float)

parser.add_argument("-seed", type=int, default=7,
					 help="Seed for random initialization")
parser.add_argument("-sigma", type=float, default=None,
					 help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]")
parser.add_argument("--embedding_dim", type=int, default=-1,
					 help="using embedding")
# parse the loss type
parser.add_argument('--loss_type', default='TOP1', type=str)
# parser.add_argument('--loss_type', default='BPR', type=str)
parser.add_argument('--topk', default=5, type=int)
# etc
parser.add_argument('--bptt', default=1, type=int)
parser.add_argument('--test_observed', default=5, type=int)
parser.add_argument('--window_size', default=30, type=int)
parser.add_argument('--warm_start', default=5, type=int)

parser.add_argument('--n_epochs', default=20, type=int)
parser.add_argument('--time_sort', default=False, type=bool)
parser.add_argument('--model_name', default='GRU4REC', type=str)
parser.add_argument('--save_dir', default='models', type=str)
parser.add_argument('--data_folder', default='../Data/movielen/1m/', type=str)
parser.add_argument('--train_data', default='train_item.pickle', type=str)
parser.add_argument('--valid_data', default='test_item.pickle', type=str)
parser.add_argument('--test_data', default='test_item.pickle', type=str)
parser.add_argument("--is_eval", action='store_true')
parser.add_argument('--load_model', default=None,  type=str)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--data_name', default=None, type=str)
parser.add_argument('--shared_embedding', default=None, type=int)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(7)

if args.cuda:
	torch.cuda.manual_seed(args.seed)

def make_checkpoint_dir():
	print("PARAMETER" + "-"*10)
	now = datetime.datetime.now()
	S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
	save_dir = os.path.join(args.checkpoint_dir, S)
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	args.checkpoint_dir = save_dir

	with open(os.path.join(args.checkpoint_dir, 'parameter.txt'), 'w') as f:
		for attr, value in sorted(args.__dict__.items()):
			print("{}={}".format(attr.upper(), value))
			f.write("{}={}\n".format(attr.upper(), value))

	print("---------" + "-"*10)

def init_model(model):
	if args.sigma is not None:
		for p in model.parameters():
			if args.sigma != -1 and args.sigma != -2:
				sigma = args.sigma
				p.data.uniform_(-sigma, sigma)
			elif len(list(p.size())) > 1:
				sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
				if args.sigma == -1:
					p.data.uniform_(-sigma, sigma)
				else:
					p.data.uniform_(0, sigma)

def count_parameters(model):
	parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("parameter_num", parameter_num) 

def save_data2pickle(data, data_dir, data_flag):
	if data_flag == "train":
		file_name = data_dir+".pickle"
		print("file name", file_name)
		f = open(file_name, "wb")
		pickle.dump(data, f)
		f.close()
	
	if data_flag == "valid":
		file_name = data_dir+".pickle"
		print("file name", file_name)
		f = open(file_name, "wb")
		pickle.dump(data, f)
		f.close()

	if data_flag == "test":
		file_name = data_dir+".pickle"
		print("file name", file_name)
		f = open(file_name, "wb")
		pickle.dump(data, f)
		f.close()

def main():
	hidden_size = args.hidden_size
	num_layers = args.num_layers
	batch_size = args.batch_size
	dropout_input = args.dropout_input
	dropout_hidden = args.dropout_hidden
	embedding_dim = args.embedding_dim
	final_act = args.final_act
	loss_type = args.loss_type
	topk = args.topk
	optimizer_type = args.optimizer_type
	lr = args.lr
	weight_decay = args.weight_decay
	momentum = args.momentum
	eps = args.eps
	BPTT = args.bptt

	n_epochs = args.n_epochs
	time_sort = args.time_sort

	window_size = args.window_size

	shared_embedding = args.shared_embedding
	print("shared_embedding", str(shared_embedding))

	if embedding_dim == -1:
		print("embedding dim not -1", embedding_dim)
		raise AssertionError()

	train_data = args.data_folder+args.train_data
	valid_data = args.data_folder+args.valid_data
	test_data = args.data_folder+args.valid_data

	print("Loading train data from {}".format(train_data))
	print("Loading valid data from {}".format(valid_data))
	print("Loading test data from {}\n".format(test_data))

	data_name = args.data_name

	print("*"*10)
	observed_threshold = args.test_observed
	print("train load")
	train_data = dataset_shiv.DatasetRNN(train_data, data_name, observed_threshold, window_size)
	print("+"*10)
	print("valid load")
	valid_data = dataset_shiv.DatasetRNN(valid_data, data_name, observed_threshold, window_size, itemmap=train_data.m_itemmap)
	test_data = dataset_shiv.DatasetRNN(test_data, data_name, observed_threshold, window_size)

	# debug_train_data_name = args.data_folder+"train_"+args.data_name+"_"+str(args.window_size)
	# debug_valid_data_name = args.data_folder+"valid_"+args.data_name+"_"+str(args.window_size)
	# debug_test_data_name = args.data_folder+"test_"+args.data_name+"_"+str(args.window_size)
	
	# save_data2pickle(train_data, debug_train_data_name, "train")
	# save_data2pickle(valid_data, debug_valid_data_name, "valid")
	# save_data2pickle(test_data, debug_test_data_name, "test")

	# exit()

	if not args.is_eval:
		make_checkpoint_dir()

	input_size = len(train_data.items)
	output_size = input_size
	print("input_size", input_size)

	train_data_loader = dataset_shiv.DataLoaderRNN(train_data, batch_size)
	
	valid_data_loader = dataset_shiv.DataLoaderRNN(valid_data, batch_size)

	# params_dataloader = {"batch_size":64, "shuffle": True, "num_workers":6}

	# train_data_loader = data.DataLoader(train_data, **params_dataloader)
	# valid_data_loader = data.DataLoader(valid_data, **params_dataloader)
	myhost = os.uname()[1]
	file_time = datetime.datetime.now().strftime('%H_%M_%d_%m')

	output_file = myhost+"_"+file_time
	output_file = output_file+"_"+str(hidden_size)+"_"+str(batch_size)+"_"+str(embedding_dim)+"_"+str(optimizer_type)+"_"+str(lr)+"_"+str(window_size)+"_"+str(shared_embedding)
	output_file = output_file+"_"+str(data_name)
	output_f = open(output_file, "w")

	if not args.is_eval:
		model = GRU4REC(window_size, input_size, hidden_size, output_size,
							final_act=final_act,
							num_layers=num_layers,
							use_cuda=args.cuda,
							batch_size=batch_size,
							dropout_input=dropout_input,
							dropout_hidden=dropout_hidden,
							embedding_dim=embedding_dim, 
							shared_embedding=shared_embedding
							)

		# init weight
		# See Balazs Hihasi(ICLR 2016), pg.7
		
		count_parameters(model)

		init_model(model)

		optimizer = Optimizer(model.parameters(),
								  optimizer_type=optimizer_type,
								  lr=lr,
								  weight_decay=weight_decay,
								  momentum=momentum,
								  eps=eps)

		loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)

		trainer = TrainerRNN(model,
							  train_data=train_data_loader,
							  eval_data=valid_data_loader,
							  optim=optimizer,
							  use_cuda=args.cuda,
							  loss_func=loss_function,
							  topk = args.topk,
							  args=args)

		trainer.train(0, n_epochs - 1, batch_size, output_f)
		output_f.close()
	else:
		if args.load_model is not None:
			print("Loading pre trained model from {}".format(args.load_model))
			checkpoint = torch.load(args.load_model)
			model = checkpoint["model"]
			model.gru.flatten_parameters()
			optim = checkpoint["optim"]
			loss_function = LossFunction(loss_type=loss_type, use_cuda=args.cuda)
			evaluation = Evaluation(model, loss_function, use_cuda=args.cuda)
			loss, recall, mrr = evaluation.evalRNN(valid_data)
			print("Final result: recall = {:.2f}, mrr = {:.2f}".format(recall, mrr))
		else:
			print("Pre trained model is None!")


if __name__ == '__main__':
	main()