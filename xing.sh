CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python main_rnn.py --data_folder ../Data/xing/ --train_data train_item.pickle --valid_data test_item.pickle --test_data test_item.pickle --data_name xing --embedding_dim 275 --hidden_size 275 --lr 0.001 --window_size 25 --test_observed 5 --n_epochs 50 --shared_embedding 1 --batch_size 275 --optimizer_type Adam --loss_type 'XE' --valid_start_time 1512172800 --test_start_time 1512259200 --negative_num 10000 --topk 20 