
## test
python test.py --name ucf101_split1_128 --gpu_ids 0 --val_list dataset/refineucf101_rgb_val_split_1.txt --root_path /home/luxiusheng/Documents/Datasets/UCF101_OF --batch_size 1 --num_segments 10 --crop_num 10

## train 
nohup python train.py --root_path /home/luxiusheng/Documents/Datasets/UCF101_OF --train_list dataset/refineucf101_rgb_train_split_1.txt --val_list dataset/refineucf101_rgb_val_split_1.txt --name ucf101_split1_128 --gpu_ids 1 --epochs 200 --lr_decay_epoch 50 --batch_size 16 --workers 4 >ucf101_split1_128.log 2>&1 &