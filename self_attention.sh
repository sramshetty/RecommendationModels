CUDA_VISIBLE_DEVICES=0 python main_self_attn.py --data_folder ../Data/tmall/100k_unknown_cate/ --data_action item_time.pickle --data_cate cate_time.pickle --data_time time_time.pickle --data_name taobao --num_heads 6 --embedding_dim 300 --hidden_size 300 --lr 0.001 --window_size 20 --test_observed 5 --n_epochs 10 --position_embedding 1 --shared_embedding 1 --batch_size 300 --optimizer_type Adam --loss_type 'XE' --valid_start_time 1512172800 --test_start_time 1512259200 --negative_num 10000 --topk 20