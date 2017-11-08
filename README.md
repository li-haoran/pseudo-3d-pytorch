# Pseudo-3D Residual Networks

This repo implements the network structure of P3D[1] with PyTorch, pre-trained model weights are converted from caffemodels, which is supported from the [author's repo](https://github.com/ZhaofanQiu/pseudo-3d-residual-networks)



### Requirements:

- pytorch
- numpy

### Structure details

In the author's official repo, only P3D-199 is released. Besides this deepest P3D-199, I also implement P3D-63 and P3D-131, which are respectively modified from ResNet50-3D and ResNet101-3D, the two nets may bring more convenience to users who have only memory-limited GPUs.


### Pretrained weights
(Pretrained weights of P3D63 and P3D131 are not yet supported) 

1, P3D Resnet trained on Kinetics dataset:

 [BaiduYun url](http://pan.baidu.com/s/1nv7Q7NF)
 
2, P3D ResNet trianed on Kinetics Optical Flow (TVL1):

 [BaiduYun url](http://pan.baidu.com/s/1nv7Q7NF)


### Example Code

    from __future__ import print_function
    from p3d_model import *
    
    model = P3D199(pretrained=True,num_classes=400)
    model = model.cuda()
    data=torch.autograd.Variable(torch.rand(10,3,16,160,160)).cuda()   # if modality=='Flow', please change the 2nd dimension 3==>2
    out=model(data)
    print (out.size(),out)
    

### Ablation settings

1. **ST-Structures**:

    All P3D models in this repo support various forms of ST-Structures like ('A','B','C') ,('A','B') and ('A'), code is as follows.

    ```
    model = P3D63(ST_struc=('A','B'))
    model = P3D131(ST_struc=('C'))
    ```
    
2. **Flow and RGB models**:
    
    Set parameter *modality='RGB'* as 'RGB' model, 'Flow' as flow model. Flow model i trained on TVL1 optical flow images.
    
    ```
    model= P3D199(pretrained=True,modality='Flow')
    ```
3. **Finetune the model**

    when finetune the models on your custom dataset, use get_optim_policies() to set different learning speed for different layers. e.g. When dataset is small, Only need to train several deepest layers, set *slow_rate=0.8* in code, and change the following *lr_mult*,*decay_mult*. 


Reference:

 [1][Learning Spatio-Temporal Representation with Pseudo-3D Residual](http://openaccess.thecvf.com/content_iccv_2017/html/Qiu_Learning_Spatio-Temporal_Representation_ICCV_2017_paper.html)