CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python main_time.py --data_folder ../Data/tmall/100k_unknown_cate/ --data_name taobao --embedding_dim 300 --hidden_size 300 --lr 0.005 --model_name 'GRU4REC'