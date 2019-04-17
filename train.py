
import time
from collections import OrderedDict
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


from opts.train_opts import TrainOptions
from data.create_dataloader import CreateDataLoader
from models.p3d_model import P3dModel
from utils.Vis import Visualizer
from utils.metrics import AverageMeter, accuracy

def main():
    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
    else:    
        start_epoch, epoch_iter = 1, 0


    print '===========training options=========='
    #print opt

    ###def model for train
    model=P3dModel()
    model.initialize(opt)
    
    ### def train continue or fineture from pretrained model
    if opt.continue_train:
        model.load(fineture=False,which_epoch=start_epoch-1,pretrain='')
    else:
        if opt.modality=='RGB':
            pretrained_file='p3d_rgb_199.checkpoint.pth.tar'
        elif opt.modality=='Flow':
            pretrained_file='p3d_flow_199.checkpoint.pth.tar'
        model.load(fineture=True,pretrain=pretrained_file)

    print '%s is useing'%(model.name())

    #def vis
    Visual=Visualizer(opt)
    ##
    dummy_input = torch.rand(1, 3, 16, 224, 224).cuda()
    Visual.tbx_write_net(model.model,dummy_input)

   
    ### def all data loader
    train_data_loader,val_data_loader, dataset_size,_= CreateDataLoader(opt,model)
    
    
    print '#training images = %d' %(dataset_size)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    cudnn.benchmark = True

    #def metrics
    top1=AverageMeter()
    top5=AverageMeter()
    losses=AverageMeter()

    total_steps = (start_epoch-1) * dataset_size + epoch_iter

    print_delta = total_steps % opt.print_freq
    update_size=opt.larger_batch_size//opt.batch_size

    update_num=0
    model.module.optimizer.zero_grad()
    for epoch in range(start_epoch, opt.epochs + 1):
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        top1.reset()
        top5.reset()
        losses.reset()
        model.train()     
        for i, data in enumerate(train_data_loader, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            #print data['data'].shape
            #print data['label'].shape

            input=Variable(data['data'].cuda())
            label=Variable(data['label'].cuda())
            ############## Forward Pass ######################
            pred,loss= model(input, label,)

            #need mean or not/
            # loss=torch.mean(loss) 

            pt1,pt5,_=accuracy(pred.data,data['label'].cuda(),topk=(1,5))
            top1.update(pt1.item(),input.size(0))
            top5.update(pt5.item(),input.size(0))
            losses.update(loss.item(),input.size(0))
            ############### Backward Pass ####################
            # update model weights
            
            loss.backward()
            update_num+=1
            if update_num==update_size:
            
                model.module.optimizer.step()
                model.module.optimizer.zero_grad()
                update_num=0
          

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {'train_loss':losses.get(),'top1':top1.get(),'top5':top5.get()}
                t = (time.time() - iter_start_time) / opt.batch_size
                Visual.print_current_errors(epoch,epoch_iter,errors,t)
                Visual.tbx_write_errors(errors,total_steps,'Train/loss')
                top1.reset()
                top5.reset()
                losses.reset()


            if epoch_iter >= dataset_size:
                break
        
        ### save model for this epoch
        ##valid here
        top1.reset()
        top5.reset()
        losses.reset()
        v__start_time=time.time()
        this_iter=0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_data_loader):

                input=Variable(data['data'].cuda())
                label=Variable(data['label'].cuda())
                
                pred,loss= model(input, label,)


                pt1,pt5,_=accuracy(pred.data,data['label'].cuda(),topk=(1,5))
                top1.update(pt1.item(),input.size(0))
                top5.update(pt5.item(),input.size(0))
                losses.update(loss.item(),input.size(0))
                this_iter=this_iter+opt.batch_size

            errors = {'valid_loss':losses.get(),'top1':top1.get(),'top5':top5.get()}
            t = (time.time() - v__start_time) / opt.batch_size
            Visual.print_current_errors(epoch,this_iter,errors,t)
            Visual.tbx_write_errors(errors,total_steps,'Valid/loss')
            top1.reset()
            top5.reset()
            losses.reset()



        if epoch % opt.save_epoch_freq == 0:
            
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch % opt.lr_decay_epoch==0:
            model.module.update_learning_rate()


if __name__ =='__main__':
    main()