CUDA_VISIBLE_DEVICES=1 python main.py --data_folder ../Data/tmall/100k_unknown_cate/ --data_action item.pickle --data_cate cate.pickle --data_time time.pickle --data_name taobao --embedding_dim 128 --hidden_size 300 --lr 0.0005 --window_size 100 --test_observed 5 --n_epochs 300 --shared_embedding 1 --batch_size 200 --optimizer_type Adam --negative_num 1000 --topk 5 --loss_type "XE"
