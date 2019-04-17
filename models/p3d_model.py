
import numpy as np
import torch
import os
import sys
from torch.autograd import Variable
from models.p3d_nets import P3D, Bottleneck

class P3dModel(torch.nn.Module):
    def name(self):
        return 'P3dModel'
    
    def initialize(self, opt):
        self.opt=opt
        modality=opt.modality
        layers=[]
        if opt.layers=='199':
            layers=[3, 8, 36, 3]
        elif opt.layers=='131':
            layers = [3, 4, 23, 3]
        elif opt.layers=='63':
            layers=[3, 4, 6, 3]
        else:
            print 'unknown layers'

        
        if opt.dataset == 'ucf101':
            self.output_channel = 101
        elif opt.dataset == 'hmdb51':
            self.output_channel = 51
        elif opt.dataset == 'kinetics':
            self.output_channel = 400
        elif opt.dataset == 'Paction':
            self.output_channel = 15
        else:
            raise ValueError('Unknown dataset '+opt.dataset)

                # some private attribute
        self.input_channel = 3 if modality=='RGB' else 2

        self.input_size=(self.input_channel,16,self.opt.input_size,self.opt.input_size)       # input of the network
        self.input_mean = [0.485, 0.456, 0.406] if modality=='RGB' else [0.5]
        self.input_std = [0.229, 0.224, 0.225] if modality=='RGB' else [np.mean([0.229, 0.224, 0.225])]
        shortcut_type=opt.shortcut_type
        dropout=opt.dropout
        ST_struc=('A','B','C')
        final_pool=5
        if self.opt.input_size==224:
            final_pool=7
        self.model = P3D(Bottleneck, layers, modality=modality,num_classes=self.output_channel,\
                    shortcut_type=shortcut_type,dropout=dropout,ST_struc=ST_struc,final_pool=final_pool)

        self.criterion=torch.nn.CrossEntropyLoss().cuda()

        print(self.model)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            self.model.cuda(opt.gpu_ids[0])
        
        if opt.isTrain:
            self.old_lr=opt.lr
            params=self.get_optim_policies(modality)
            self.optimizer = torch.optim.SGD(params,
                                    opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    @property
    def scale_size(self):
        return self.input_size[2] * self.opt.scale_size // self.opt.input_size   # asume that raw images are resized (340,256).
    @property
    def num_classes(self):
        return self.output_channel


    @property
    def temporal_length(self):
        return self.input_size[1]

    @property
    def crop_size(self):
        return self.input_size[2]


    # custom operation
    def get_optim_policies(self,modality,enable_pbn=True):
        '''
        first conv:         weight --> conv weight
                            bias   --> conv bias
        normal action:      weight --> non-first conv + fc weight
                            bias   --> non-first conv + fc bias
        bn:                 the first bn2, and many all bn3.
        '''
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m,torch.nn.BatchNorm2d):
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        slow_rate=self.opt.slow_rate
        n_fore=int(len(normal_weight)*slow_rate)
        slow_feat=normal_weight[:n_fore] # finetune slowly.
        slow_bias=normal_bias[:n_fore] 
        normal_feat=normal_weight[n_fore:]
        normal_bias=normal_bias[n_fore:]

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if modality == 'Flow' else 1*self.opt.slow_lr_mult, 'decay_mult': 1,
            'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if modality == 'Flow' else 2*self.opt.slow_lr_mult, 'decay_mult': 0,
            'name': "first_conv_bias"},
            {'params': slow_feat, 'lr_mult': 1*self.opt.slow_lr_mult, 'decay_mult': 1,
            'name': "slow_feat"},
            {'params': slow_bias, 'lr_mult': 2*self.opt.slow_lr_mult, 'decay_mult': 0,
            'name': "slow_bias"},
            {'params': normal_feat, 'lr_mult': 1 , 'decay_mult': 1,
            'name': "normal_feat"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult':0,
            'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1*self.opt.slow_bn_mult, 'decay_mult': 0,
            'name': "BN scale/shift"},
        ]

    def forward(self,x,y):
        
        output=self.model.forward(x)
        loss=self.criterion(output,y)
        return [output,loss]

    def inference(self, x):
        # Encode Inputs        
        return self.model.forward(x)


    def save(self, which_epoch):

        save_filename = 'net_%s_%04d.pth.tar' % (self.name(),which_epoch)
        save_path = os.path.join(self.opt.exp_dir, save_filename)
        torch.save(self.model.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.model.cuda()


    def load(self,fineture=True,which_epoch=0,pretrain=''):
        if not fineture:
            save_filename = 'net_%s_%04d.pth.tar' % (self.name(),which_epoch)
            save_path = os.path.join(self.opt.exp_dir, save_filename)  
            if not os.path.isfile(save_path):
                print('%s not exists yet!' % save_path)
            else:
                self.model.load_state_dict(torch.load(save_path))
               
        else:
            pretrained = torch.load(pretrain)
            print '======pretrained_model========'
            print pretrained.keys(),pretrained['arch']
            arch=pretrained['arch']

            pretrained_dict=pretrained['state_dict']
            model_dict = self.model.state_dict()
            #print '=========this model==========='
            #print model_dict.keys()
            for k, v in pretrained_dict.items():                      
                if v.size() == model_dict[k].size():
                    model_dict[k] = v

            if sys.version_info >= (3,0):
                not_initialized = set()
            else:
                from sets import Set
                not_initialized = Set()                    

            for k, v in model_dict.items():
                if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                    not_initialized.add(k.split('.')[0])
            
            print(sorted(not_initialized))
            self.model.load_state_dict(model_dict)         


    def update_learning_rate(self):
        lr = self.old_lr *self.opt.lr_decay_ratio   

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
        
