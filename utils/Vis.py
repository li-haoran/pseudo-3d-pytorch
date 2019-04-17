### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import ntpath
import time
import scipy.misc
from tensorboardX import SummaryWriter
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.name = opt.name
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
        self.writer=SummaryWriter(self.log_dir)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    # errors: same format as |errors| of plotCurrentErrors
    def tbx_write_errors(self,errors,total_steps,group='Train/loss'):
        self.writer.add_scalars(group, errors, total_steps)

    def tbx_write_net(self,model,input):  
        self.writer.add_graph(model,input)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items(): 
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)


def plot_mix(cmp,opt,with_num=False):
 
    class_name=[s.split(' ')[1].strip('\r\n') for s in open(opt.classind,'r') ]

    fig, ax = plt.subplots()
    im = ax.imshow(cmp)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_name)))
    ax.set_yticks(np.arange(len(class_name)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_name,fontsize='xx-small')
    ax.set_yticklabels(class_name,fontsize='xx-small')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    log_name='%s_nseg_%d_ncrop_%d_nepoch_%d.txt'%(opt.exp_dir,opt.num_segments,opt.crop_num,opt.epoch_num)

    with open(log_name, "a") as log_file:
            log_file.write('ucf_101 result\n' )
            message='         | '
            for name in class_name:
                message+='%s |'%(name)
            log_file.write('%s\n' % message)
            for j,name in enumerate(class_name):
                message='%s | '%(name)
                for i in range(len(class_name)):
                    message+='%.3f |'%(cmp[j,i])
                log_file.write('%s \n' % message)

    #find_top5
    index=np.argsort(cmp,axis=1)
    index=index[:,::-1]
    with open(log_name, "a") as log_file:
            log_file.write('ucf_101 result_top5\n' )
            for j,name in enumerate(class_name):
                message='%s : \n'%(name)
                for i in range(5):
                    message+=' | %.3f : %s |\n'%(cmp[j,index[j,i]],class_name[index[j,i]])
                log_file.write('%s ' % message)


    

    # Loop over data dimensions and create text annotations.
    if with_num:
        for i in range(len(class_name)):
            for j in range(len(class_name)):
                text = ax.text(j, i, cmp[i, j],
                            ha="center", va="center", color="w")

    ax.set_title("%s result"%(opt.dataset))
    fig.tight_layout()
    plt.savefig('%s_nseg_%d_ncrop_%d_nepoch_%d.pdf'%(opt.exp_dir,opt.num_segments,opt.crop_num,opt.epoch_num),dpi=100)
    plt.close()

