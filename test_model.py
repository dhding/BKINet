import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import argparse
import time
import random
import json
import math
from distutils.version import LooseVersion
from xml.sax.handler import feature_external_ges
import scipy.misc
import logging
import datetime
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.data_loader import *
from model.base_model import *
from utils.losses import *
from utils.parsing_metrics import *
from utils.utils import *
from utils.checkpoint import save_checkpoint, load_pretrain, load_resume

from tensorboardX import SummaryWriter
import copy

#import apex.amp as amp
from torch.cuda.amp import autocast as autocast, GradScaler


def main():
    parser = argparse.ArgumentParser(description='Dataloader test')
    parser.add_argument('--gpu', default='2', help='gpu id')
    parser.add_argument('--workers', default=8, type=int, help='num workers for data loading')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--clip_model', default='RN50', type=str, help='clip model: R50 R101')
    parser.add_argument('--nb_epoch', default=50, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.000025, type=float, help='learning rate')
    parser.add_argument('--power', default=0.1, type=float, help='lr poly power')
    parser.add_argument('--steps', default=[20, 30, 40], type=list, help='in which step lr decay by power')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--size', default=416, type=int, help='image size')
    parser.add_argument('--dataset', default='refcoco', type=str,
                        help='refcoco/refcoco+/refcocog')
    
    parser.add_argument('--num_query', default=16, type=int, help='the number of query')
    parser.add_argument('--w_seg', default=0.1, type=float, help='weight of the seg loss')
    parser.add_argument('--w_coord', default=5, type=float, help='weight of the reg loss')
    parser.add_argument('--tunelang', dest='tunelang', default=True, action='store_true', help='if finetune language model')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='../BKINet/ln_data',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='../BKINet/data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--time', default=15, type=int,
                        help='maximum time steps (lang length) per batch')

    parser.add_argument('--fusion_dim', default=1024, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='saved_models/model_refcoco_batch0_model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str, help='Name head for saved model')
    
    parser.add_argument('--seg_thresh', default=0.35, type=float, help='seg score above this value means foreground')
    parser.add_argument('--seg_out_stride', default=2, type=int, help='the seg out stride')
    parser.add_argument('--best_iou', default=-float('Inf'), type=int, help='the best accu')

    global args, anchors_full, writer
    args = parser.parse_args()
    args.gsize = 32 

    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
  
    ## save logs
    if args.savename=='default':
        args.savename = 'model_%s_batch%d'%(args.dataset, args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.INFO, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(str(sys.argv))
    logging.info(str(args))

    writer = SummaryWriter(comment=args.savename)

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time, # 15
                         augment=True)
    
    val_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='val',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)

    if args.dataset == 'refcocog':
    
        test_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='test',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)
    
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    
    else:
    
        testA_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='testA',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)
        testB_dataset = ReferDataset(data_root=args.data_root,
                         dataset=args.dataset,
                         split_root=args.split_root,
                         split='testB',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time)
    

    
        testA_loader = DataLoader(testA_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
        testB_loader = DataLoader(testB_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    

    ## Model
    model = Model(clip_model=args.clip_model, tunelang=args.tunelang, num_query=args.num_query)
    
    args.start_epoch = 0
    if args.pretrain:
        model=load_pretrain(model,args,logging)

    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    visu_param = [param for name, param in model.named_parameters() if 'visumodel' in name]
    text_param = [param for name, param in model.named_parameters() if 'textmodel' in name]
    rest_param = [param for name, param in model.named_parameters() if 'textmodel' not in name and 'visumodel' not in name]

    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in text_param])
    sum_fusion = sum([param.nelement() for param in rest_param])
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    ## optimizer; adam default
    if args.tunelang:
        optimizer = torch.optim.Adam([{'params': rest_param, 'lr': args.lr},
                                      {'params': visu_param, 'lr': args.lr / 10.},
                                      {'params': text_param, 'lr': args.lr / 10.}])
    else:
        optimizer = torch.optim.Adam([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr / 10.}], lr=args.lr)

    # Initialization
    scaler = GradScaler()
    model = model.cuda()

    best_miou_seg = -float('Inf')
    if args.resume:
        model = load_resume(model, optimizer, args, logging)
        best_miou_seg = args.best_iou
        print(best_miou_seg)
    
    if args.dataset == 'refcocog':
        print('\nTest testing:')
        miou_seg, prec = validate_epoch(test_loader, model, 'Test')
    
    else:
        print('\nTestA testing:')
        miou_seg, prec = validate_epoch(testA_loader, model, 'TestA')
    
    
        print('\nTestB testing:')
        miou_seg, prec = validate_epoch(testB_loader, model, 'TestB')

def validate_epoch(val_loader, model, mode='val'):
    batch_time = AverageMeter()
    miou = AverageMeter()
    miou_seg = AverageMeter()

    prec=dict()
    thresholds = np.arange(0.5, 1, 0.05)

    for thresh in thresholds:
        prec[thresh]= AverageMeter()

    model.eval()
    end = time.time()
    idx = 0

    t_all = []

    for batch_idx, (imgs, word_id, word_mask, bbox, seg_map, ratio, dw, dh, im_id, phrase, draw_img) in enumerate(val_loader):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()

        seg_map = seg_map.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        seg_map = Variable(seg_map)

        t1 = time.time()
        with torch.no_grad():
            mask_out = model(image, word_id, word_mask)
            mask_out = mask_out.sigmoid()
        t2 = time.time()
        t_all.append(t2-t1)

        ## test: convert pred, gt box to original scale with meta-info
        ih = seg_map.shape[-2]
        iw = seg_map.shape[-1]
        nh = int(ih * ratio)
        nw = int(iw * ratio)
        top, bottom = int(dh[0]), nh + int(dh[0])
        left, right = int(dw[0]), nw + int(dw[0])
        ratio = float(ratio)
        new_shape = (iw, ih)
        
        ## revert image for visualization
        seg_map_np = seg_map[0,:,:,:].data.cpu().numpy().transpose(1,2,0)
        seg_map_np = cv2.resize(seg_map_np, new_shape, interpolation=cv2.INTER_CUBIC)
        img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)
        img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
        if idx % 5 == 0:    
            print_img = img_np.copy()
            print_seg = img_np.copy()
            print_seg_t = img_np.copy()
        img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))
        
        # seg
        mask_out = mask_out[0].data.cpu().numpy().transpose(1,2,0)
        mask_out = cv2.resize(mask_out, (args.size, args.size))
        mask_out_np = mask_out[top:bottom, left:right]
        mask_out_np = cv2.resize(mask_out_np, new_shape)
        seg_iou, seg_prec = cal_seg_iou(seg_map[0].cpu().numpy(), mask_out_np, args.seg_thresh)
        miou_seg.update(seg_iou, imgs.size(0))
        for thresh in thresholds:
            prec[thresh].update(seg_prec[thresh], imgs.size(0))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 1000 == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'seg_iu {seg.val:.4f} ({seg.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, seg=miou_seg)
            print(print_str)
            logging.info(print_str)  
        idx = idx + 1

    print(miou_seg.avg)
    for thresh in thresholds:
            print("prec@%f: %f"%(thresh,float(prec[thresh].avg)))
            logging.info("prec@%f:%f"%(thresh,float(prec[thresh].avg)))
    logging.info("%f,%f"%(float(miou.avg), miou_seg.avg))
    return miou_seg.avg, prec


if __name__ == "__main__":
    main()
