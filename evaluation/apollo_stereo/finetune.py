from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--batchsize', type=int ,default=3,
                    help='batchsize for training')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= args.batchsize, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 8, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
    
# update pretrained model params according to my model params
def update_model(my_model, pretrained_dict):
    my_model_dict = my_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict}
    # 2. overwrite entries in the existing state dict
    my_model_dict.update_disp_metrics(pretrained_dict)
    
    return my_model_dict

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(update_model(model, state_dict['state_dict']))

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

from logger import Logger
def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
loogger_path = os.path.join(args.savemodel,'tb_log')
print(loogger_path)
logger = Logger(loogger_path)

def train(imgL,imgR,disp_L):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        #---------
        mask = (disp_true > 0)
        mask.detach_()
        #----

        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
#            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
            loss =  0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + \
                    0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + \
                    F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output = model(imgL,imgR)
            output = torch.squeeze(output3,1)
            loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data[0]

def test(imgL,imgR,disp_true, batch_idx, epoch):
        model.eval()
        tb_imgL = imgL
        imgL   = Variable(torch.FloatTensor(imgL), volatile=True)
        imgR   = Variable(torch.FloatTensor(imgR), volatile=True) 
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)+(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()
        
        info = {'left_img': tb_imgL.view(-1, 3, 368, 1232)[:1],
                'disp_gt':true_disp.view(-1, 368, 1232)[:1],
                'disp_pred':pred_disp.view(-1, 368, 1232)[:1]
                }
        for tag, images in info.items():
            logger.image_summary('epoch_'+str(epoch)+'/'+tag, images, batch_idx)
        return float(torch.sum(correct))/float(len(index[0]))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= args.epochs/5:
       lr = 0.001
    elif epoch <= 2 * args.epochs/5:
       lr = 0.0005
    elif epoch <= 3 * args.epochs/5:
       lr = 0.0002
    elif epoch <= 4 * args.epochs/5:
       lr = 0.0001
    else:
       lr = 0.00005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc=0
    max_epo=0
    start_full_time = time.time()
    log_file_name = os.path.join(args.savemodel, 'log_ft.txt')
    
    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)
        
        #trainning
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time() 
            
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        
        # testing
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL,imgR, disp_L, batch_idx, epoch)
            print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
            total_test_loss += test_loss
        
        with open(log_file_name, 'a') as f:
            print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100), file=f)
            if epoch % 20 == 0:
                savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'train_loss': total_train_loss/len(TrainImgLoader),
                            'test_loss': total_test_loss/len(TestImgLoader)*100,}, savefilename)
            
            if total_test_loss/len(TestImgLoader)*100 > max_acc:
                max_acc = total_test_loss/len(TestImgLoader)*100
                max_epo = epoch
                savefilename = args.savemodel+'finetune_best.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                    'test_loss': total_test_loss/len(TestImgLoader)*100,
                    }, savefilename)
            print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc),file=f)
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)
    


if __name__ == '__main__':
   main()