CUDA_VISIBLE_DEVICES=0 python main_self_attn.py --data_folder ../Data/tmall/100k_unknown_cate/ --data_name taobao --num_heads 6 --embedding_dim 20 --hidden_size 20 --lr 0.001 --window_size 20 --test_observed 5 --n_epochs 10 --position_embedding 1 --shared_embedding 1 --batch_size 20 --optimizer_type Adam --loss_type 'XE' --topk 20 