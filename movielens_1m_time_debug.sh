CUDA_VISIBLE_DEVICES=0 python main_time_movielens.py --data_folder ../Data/ctr/ --data_action movielen_1M_time --data_name movielens --embedding_dim 64 --hidden_size 64 --lr 0.01 --test_observed 5 --n_epochs 100 --shared_embedding 1 --batch_size 300 --optimizer_type Adam --loss_type 'XE' --negative_num 1000 --topk 20 