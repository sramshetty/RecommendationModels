import pandas as pd
import numpy as np
import torch

import pickle
import random
# import sys

class Dataset(object):

	def __init__(self, action_file, observed_threshold, window_size, itemmap=None):
		action_f = open(action_file, "rb")

		self.m_itemmap = {}

		action_seq_arr_total = None
		action_seq_arr_total = pickle.load(action_f)

		seq_num = len(action_seq_arr_total)
		print("seq num", seq_num)

		self.m_x_short_action_list = []
		self.m_x_short_actionNum_list = []
		self.m_y_action = []
		self.m_y_action_idx = []

		print("loading item map")

		print("finish loading item map")
		print("observed_threshold", observed_threshold, window_size)
		print("loading data")
		for seq_index in range(seq_num):
			action_seq_arr = action_seq_arr_total[seq_index]

			action_num_seq = len(action_seq_arr)

			if action_num_seq < window_size :
				window_size = action_num_seq

			for action_index in range(action_num_seq):
				item = action_seq_arr[action_index]
				if item not in self.m_itemmap:
					item_id = len(self.m_itemmap)
					self.m_itemmap[item] = item_id

				if action_index < observed_threshold:
					continue

				if action_index <= window_size:
					input_sub_seq = action_seq_arr[:action_index]
					
					target_sub_seq = action_seq_arr[action_index]
					self.m_x_short_action_list.append(input_sub_seq)
					self.m_x_short_actionNum_list.append(len(input_sub_seq))
					self.m_y_action.append(target_sub_seq)
					self.m_y_action_idx.append(action_index)

				if action_index > window_size:
					input_sub_seq = action_seq_arr[action_index-window_size:action_index]
				
					target_sub_seq = action_seq_arr[action_index]
					self.m_x_short_action_list.append(input_sub_seq)
					self.m_x_short_actionNum_list.append(len(input_sub_seq))
					self.m_y_action.append(target_sub_seq)
					self.m_y_action_idx.append(action_index)

	@property
	def items(self):
		# print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap

class DataLoader():
	def __init__(self, dataset, batch_size):
		self.m_dataset = dataset
		self.m_batch_size = batch_size

		sorted_data = sorted(zip(self.m_dataset.m_x_short_action_list, self.m_dataset.m_x_short_actionNum_list, self.m_dataset.m_y_action, self.m_dataset.m_y_action_idx), reverse=True)

		self.m_dataset.m_x_short_action_list,self.m_dataset.m_x_short_actionNum_list , self.m_dataset.m_y_action, self.m_dataset.m_y_action_idx = zip(*sorted_data)

		input_seq_num = len(self.m_dataset.m_x_short_action_list)
		batch_num = int(input_seq_num/batch_size)
		print("seq num", input_seq_num)
		print("batch size", self.m_batch_size)
		print("batch_num", batch_num)

		x_short_action_list = [self.m_dataset.m_x_short_action_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		
		x_short_actionNum_list = [self.m_dataset.m_x_short_actionNum_list[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
		y_action = [self.m_dataset.m_y_action[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]
	
		y_action_idx = [self.m_dataset.m_y_action_idx[i*batch_size:(i+1)*batch_size] for i in range(batch_num)]

		temp = list(zip(x_short_action_list, x_short_actionNum_list, y_action, y_action_idx))

		self.m_temp = temp
	
	def __iter__(self):
		print("shuffling")

		temp = self.m_temp
		random.shuffle(temp)

		x_short_action_list, x_short_actionNum_list, y_action, y_action_idx = zip(*temp)

		batch_size = self.m_batch_size
		
		batch_num = len(x_short_action_list)

		for batch_index in range(batch_num):

			x_short_action_list_batch = x_short_action_list[batch_index]
			# x_short_cate_list_batch = x_short_cate_list[batch_index]
			x_short_actionNum_list_batch = x_short_actionNum_list[batch_index]

			y_action_batch = y_action[batch_index]
			# y_cate_batch = y_cate[batch_index]
			y_action_idx_batch = y_action_idx[batch_index]

			x_short_action_batch = []
			# x_short_cate_batch = []
			x_short_actionNum_batch = []

			mask_short_action_batch = []

			max_short_actionNum_batch = max(x_short_actionNum_list_batch)
			
			for seq_index_batch in range(batch_size):
				x_short_actionNum_seq = x_short_actionNum_list_batch[seq_index_batch]
				
				pad_x_short_action_seq = x_short_action_list_batch[seq_index_batch]+[0]*(max_short_actionNum_batch-x_short_actionNum_seq)
				x_short_action_batch.append(pad_x_short_action_seq)

			x_short_action_batch = np.array(x_short_action_batch)
			# x_short_cate_batch = np.array(x_short_cate_batch)

			x_short_actionNum_batch = np.array(x_short_actionNum_list_batch)
			mask_short_action_batch = np.arange(max_short_actionNum_batch)[None, :] < x_short_actionNum_batch[:, None]

			y_action_batch = np.array(y_action_batch)
			# y_cate_batch = np.array(y_cate_batch)
			y_action_idx_batch = np.array(y_action_idx_batch)

			x_short_action_batch_tensor = torch.from_numpy(x_short_action_batch)
			# x_short_cate_batch_tensor = torch.from_numpy(x_short_cate_batch)
			mask_short_action_batch_tensor = torch.from_numpy(mask_short_action_batch*1).float()

			y_action_batch_tensor = torch.from_numpy(y_action_batch)
			# y_cate_batch_tensor = torch.from_numpy(y_cate_batch)

			y_action_idx_batch_tensor = torch.from_numpy(y_action_idx_batch)

	
			pad_x_short_actionNum_batch = np.array([i-1 if i > 0 else 0 for i in x_short_actionNum_batch])

			yield x_short_action_batch_tensor, mask_short_action_batch_tensor, pad_x_short_actionNum_batch, y_action_batch_tensor, y_action_idx_batch_tensor


class DatasetAttn(object):

	def __init__(self, itemFile, data_name, observed_threshold, window_size, itemmap=None):
		data_file = open(itemFile, "rb")

		action_seq_arr_total = None
		data_seq_arr = pickle.load(data_file)

		if data_name == "movielen_itemmap":
			action_seq_arr_total = data_seq_arr['action_list']
			itemmap = data_seq_arr['itemmap']

		if data_name == "movielen":
			action_seq_arr_total = data_seq_arr

		if data_name == "xing":
			action_seq_arr_total = data_seq_arr

		if data_name == "taobao":
			action_seq_arr_total = data_seq_arr

		seq_num = len(action_seq_arr_total)
		print("seq num", seq_num)

		seq_len_list = []

		self.m_itemmap = itemmap
		if itemmap is None:
			self.m_itemmap = {}
		self.m_itemmap['<PAD>'] = 0

		self.m_seq_list = []

		self.m_input_action_seq_list = []
		self.m_target_action_seq_list = []
		self.m_input_seq_idx_list = []

		print("loading item map")

		for seq_index in range(seq_num):
			action_seq_arr = action_seq_arr_total[seq_index]

			action_num_seq = len(action_seq_arr)

			action_seq_list = []

			for action_index in range(action_num_seq):
				item = action_seq_arr[action_index]

				if itemmap is None: 
					if item not in self.m_itemmap:
						item_id = len(self.m_itemmap)
						self.m_itemmap[item] = item_id
				else:
					if item not in self.m_itemmap:
						continue

				item_id = self.m_itemmap[item]

				action_seq_list.append(item_id)

			self.m_seq_list.append(action_seq_list)

		print("finish loading item map")

		print("loading data")
		for action_seq_arr in self.m_seq_list:

			action_num_seq = len(action_seq_arr)
			
			if action_num_seq < window_size :
				window_size = action_num_seq

			for action_index in range(observed_threshold, window_size):
				input_sub_seq = action_seq_arr[:action_index]
				target_sub_seq = action_seq_arr[action_index]
				self.m_input_action_seq_list.append(input_sub_seq)
				self.m_target_action_seq_list.append(target_sub_seq)
				self.m_input_seq_idx_list.append(action_index)

			for action_index in range(window_size, action_num_seq):
				input_sub_seq = action_seq_arr[action_index-window_size+1:action_index]
				target_sub_seq = action_seq_arr[action_index]
				self.m_input_action_seq_list.append(input_sub_seq)
				self.m_target_action_seq_list.append(target_sub_seq)
				self.m_input_seq_idx_list.append(action_index)

	def __len__(self):
		return len(self.m_input_action_seq_list)

	def __getitem__(self, index):
		x = self.m_input_action_seq_list[index]
		y = self.m_target_action_seq_list[index]

		x = np.array(x)
		y = np.array(y)

		x_tensor = torch.LongTensor(x)
		y_tensor = torch.LongTensor(y)

		return x_tensor, y_tensor

	@property
	def items(self):
		print("first item", self.m_itemmap['<PAD>'])
		return self.m_itemmap


class DataLoaderAttn():
		def __init__(self, dataset, batch_size):
			self.m_dataset = dataset
			self.m_batch_size = batch_size

		def __iter__(self):

			batch_size = self.m_batch_size
			input_action_seq_list = self.m_dataset.m_input_action_seq_list
			target_action_seq_list = self.m_dataset.m_target_action_seq_list
			input_seq_idx_list = self.m_dataset.m_input_seq_idx_list

			input_num = len(input_action_seq_list)

			batch_num = int(input_num/batch_size)

			for batch_index in range(batch_num):
				x_batch = []
				y_batch = []
				idx_batch = []

				for seq_index_batch in range(batch_size):
					seq_index = batch_index*batch_size+seq_index_batch

					x_batch.append(input_action_seq_list[seq_index])
					y_batch.append(target_action_seq_list[seq_index])
					idx_batch.append(input_seq_idx_list[seq_index])

				x_batch, y_batch, idx_batch = self.batchifyData(x_batch, y_batch, idx_batch)

				x_batch_tensor = torch.LongTensor(x_batch)
				y_batch_tensor = torch.LongTensor(y_batch)
				idx_batch_tensor = torch.LongTensor(idx_batch)

				yield x_batch_tensor, y_batch_tensor, idx_batch_tensor

				### apply padding to seq in same batch
				### sort seq by length (or shuffle)
		def batchifyData(self, input_action_seq_batch, target_action_seq_batch, idx_batch):
			longest_len_batch = max([len(seq_i) for seq_i in input_action_seq_batch])
			batch_size = len(input_action_seq_batch)

			pad_input_action_seq_batch = np.zeros((batch_size, longest_len_batch))
			pad_target_action_seq_batch = np.zeros(batch_size)
			pad_idx_batch = np.zeros(batch_size)
			
			zip_batch = sorted(zip(idx_batch, input_action_seq_batch, target_action_seq_batch), reverse=True)

			for seq_i, (seq_idx, input_action_seq_i, target_action_seq_i) in enumerate(zip_batch):
				pad_input_action_seq_batch[seq_i, 0:len(input_action_seq_i)] = input_action_seq_i
				pad_target_action_seq_batch[seq_i] = target_action_seq_i
				pad_idx_batch[seq_i] = seq_idx

			### map item id back to start from 0
			# target_action_seq_batch = [target_i-1 for target_i in target_action_seq_batch]

			return pad_input_action_seq_batch, pad_target_action_seq_batch, pad_idx_batch
