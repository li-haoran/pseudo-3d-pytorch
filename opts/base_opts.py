### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='ucf101rgb', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--checkpoints_dir', type=str, default='experiments', help='dir to save experiments')

        self.parser.add_argument('--modality', type=str, default='RGB', help='model input modality')  
        self.parser.add_argument('--layers', type=str, default='199', help='model num of layers')
        self.parser.add_argument('--shortcut_type', type=str, default='B', help='model short cut type')
        self.parser.add_argument('--input_size', type=int, default=160, help='input_size for video')
        
        self.parser.add_argument('--dataset', type=str, default='ucf101', help='the dataset for training')
        self.parser.add_argument('--root_path', type=str, default='ucf101', help='the dataset dir')
        self.parser.add_argument('--train_list', type=str, default='train.list', help='training list video')
        self.parser.add_argument('--val_list', type=str, default='val.list', help='valid lst video')
        self.parser.add_argument('--num_segments', type=int, default=1, help='segments per videos')
        self.parser.add_argument('--scale_size', type=int, default=256, help='original input size for videos')
        self.parser.add_argument('--sample_rate', type=int, default=1, help='image ssample rate from video')

        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--workers', type=int, default=4, help='workers for dataloaders')      
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout')


        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.exp_dir=expr_dir
        if not os.path.exists(expr_dir):
            os.mkdir(expr_dir)
            
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
