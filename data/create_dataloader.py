import torch 
import torchvision
from data.dataset import TSNDataSet

from data.transforms import *


def CreateDataLoader(opt,model):
    data_length=16 if opt.modality == 'RGB' else 5
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    sample_rate=opt.sample_rate
    # Data loading code
    if opt.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    #for 224 
    scales_ratio=[1, .875, .75, ]#.66]## elimate the multi scale
    #
    #GroupScale(int(scale_size)),
    #GroupRandomCrop(crop_size),
    

    train_augmentaion= [GroupMultiScaleCrop(crop_size, scales_ratio),
                        GroupRandomHorizontalFlip(is_flow=False),
                        Stack(False),
                        ToTorchFormatTensor(True),
                        normalize,]

    val_augmentation= [GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(False),
                       ToTorchFormatTensor(True),
                       normalize,]

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(opt.root_path, opt.train_list, num_segments=opt.num_segments,
                    new_length=data_length,
                    sample_rate=sample_rate,
                    modality=opt.modality,
                    image_tmpl="{:06d}.jpg" ,
                    transform=torchvision.transforms.Compose(train_augmentaion),),
                    batch_size=opt.batch_size, shuffle=True,
                    num_workers=opt.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(opt.root_path, opt.val_list, num_segments=opt.num_segments,
                   new_length=data_length,
                   sample_rate=sample_rate,
                   modality=opt.modality,
                   image_tmpl="{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose(val_augmentation),),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    return train_loader,val_loader,len(train_loader.dataset),len(val_loader.dataset)


def CreateTestLoader(opt,model):
    data_length=16 if opt.modality == 'RGB' else 5
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    sample_rate=opt.sample_rate
    # Data loading code
    if opt.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
    if opt.crop_num == 1:
        cropping = [GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
        ]
    elif opt.crop_num == 10:
        cropping = [GroupOverSample(crop_size, int(scale_size)),
        ]
    else:
        raise ValueError("Only 1 and 10 crops are supported while we got {}".format(opt.crop_num))

    val_augmentation= cropping+[Stack(False),
                       ToTorchFormatTensor(True),
                       normalize,]

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(opt.root_path, opt.val_list, num_segments=opt.num_segments,
                   new_length=data_length,
                   sample_rate=sample_rate,
                   modality=opt.modality,
                   image_tmpl="img_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose(val_augmentation),
                   ),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    return val_loader,len(val_loader.dataset)
