CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python main_rnn.py --data_folder ../Data/xing/ --data_name xing --embedding_dim 275 --hidden_size 275 --lr 0.001 --window_size 25 --test_observed 5 --n_epochs 5 --shared_embedding 1 --batch_size 275 --optimizer_type Adam --loss_type 'XE' --topk 20 