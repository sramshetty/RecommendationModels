CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python main_self_attn.py --data_folder ../Data/tmall/100k_unknown_cate/ --data_name taobao --num_heads 1 --embedding_dim 16 --hidden_size 16 --lr 0.1 --window_size 16 --test_observed 5 --n_epochs 1 --position_embedding 1 --shared_embedding 1 --batch_size 32 --optimizer_type Adam --loss_type 'XE' --topk 5 --model_name 'self-attention'