
import time
from collections import OrderedDict
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


from opts.test_opts import TestOptions
from data.create_dataloader import CreateTestLoader
from models.p3d_model import P3dModel
from utils.Vis import plot_mix
from utils.metrics import AverageMeter, accuracy

def main():
    opt = TestOptions().parse(False)
    channel=3
    if opt.modality == 'RGB':
        channel=3
        data_length=16
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
       
    ###def model for train
    model=P3dModel()
    model.initialize(opt)
    num_classes=model.num_classes
    which_epoch=start_epoch-1
    if opt.epoch_num>0:
        which_epoch=opt.epoch_num
    model.load(fineture=False,which_epoch=which_epoch,pretrain='')

    print '%s is useing'%(model.name())
   
    ### def all data loader
    test_data_loader, dataset_size= CreateTestLoader(opt,model)
    
    
    print '#testing images = %d' %(dataset_size)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    cudnn.benchmark = True

   

    #def metrics
    top1=AverageMeter()
    top5=AverageMeter()
    class_top1=[AverageMeter() for i in range(num_classes)]
    class_top5=[AverageMeter() for i in range(num_classes)]
    mix_m=np.zeros((num_classes,num_classes),np.float32)
    top1.reset()
    top5.reset()
   
    for i in range(num_classes):
        class_top1[i].reset()
        class_top5[i].reset()

    v__start_time=time.time()
    this_iter=0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            print '%d video isprocessing'%(i)
            input=Variable(data['data'].cuda())
            label=Variable(data['label'].cuda())

            sizes=input.size()
            assert sizes[2]==opt.crop_num*opt.num_segments*data_length,'shape error'

            input=input.view(sizes[0],sizes[1],opt.crop_num*opt.num_segments,data_length,sizes[3],sizes[4])
            input=input.permute(0,2,1,3,4,5)
            input=input.view(sizes[0]*opt.crop_num*opt.num_segments,sizes[1],data_length,sizes[3],sizes[4])
            pred=model.module.inference(input)
            pred=pred.view(opt.batch_size,opt.crop_num*opt.num_segments,-1)
            new_pred=torch.sum(pred.data,1,False)


            pt1,pt5,acc_v=accuracy(new_pred,data['label'].cuda(),topk=(1,5))
            top1.update(pt1.item(),opt.batch_size)
            top5.update(pt5.item(),opt.batch_size)

            d=data['label'].numpy()
            assert d.shape[0]==1,'only support batch size ==1'
            key=d[0]
            class_top1[key].update(pt1.item(),opt.batch_size)
            class_top5[key].update(pt5.item(),opt.batch_size)

            cc=acc_v[0]
            mix_m[key,cc]+=1
            this_iter=this_iter+opt.batch_size

            #if i>100:
             #   break
        
        t = (time.time() - v__start_time) / opt.batch_size
        print '%f times test result top5: %f, top1: %f'%(t,top5.get(),top1.get())

        total_loss_file='%s/total_loss.txt'%(opt.checkpoints_dir)
        message='%s_nseg_%d_ncrop_%d_nepoch_%d : '%(opt.name,opt.num_segments,opt.crop_num,opt.epoch_num)

        with open(total_loss_file, "a") as tlf:
            message+='%f times test result top5: %f, top1: %f \n'%(t,top5.get(),top1.get())
            tlf.write('%s ' % message)
            

        print '==============each class accuracy========================='
        for i in range(num_classes):
            print 'class %d test result top5: %f, top1: %f'%(i,class_top5[i].get(),class_top1[i].get())
            mix_m[i,:]/=class_top1[i].count

        plot_mix(mix_m,opt)

        top1.reset()
        top5.reset()
        for i in range(num_classes):
            class_top1[i].reset()
            class_top5[i].reset()



if __name__ =='__main__':
    main()