CUDA_VISIBLE_DEVICES=0 python main_time.py --data_folder ../Data/BeerAdvocate/ --data_action item_time.pickle --data_cate cate_time.pickle --data_time time_time.pickle --data_name taobao --embedding_dim 32 --hidden_size 32 --lr 0.0001 --window_size 50 --test_observed 5 --n_epochs 300 --shared_embedding 1 --batch_size 100 --optimizer_type Adam --loss_type 'XE' --valid_start_time 1296185247 --test_start_time 1310947622 --negative_num 1000 --topk 20 --weight_decay 0.0001

# trainTime_threshold = 1296185247
# validTime_threshold = 1310947622
