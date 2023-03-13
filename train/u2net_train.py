import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
import numpy as np
import glob  
from copy  import deepcopy
from Inference_Mlu.data_loader import RescaleT
from Inference_Mlu.data_loader import RandomCrop
from Inference_Mlu.data_loader import ToTensorLab
from Inference_Mlu.data_loader import SalObjDataset
from Inference_Mlu.model.u2net import U2NETP
import os
import torch_mlu
import torch_mlu.core.mlu_quantize as mlu_quantize
import torch_mlu.core.mlu_model as ct
import argparse
# from tensorboardX import SummaryWriter
ct.set_cnml_enabled(False)
deivce = torch.device('mlu')

# ------- 1. define loss function --------

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.cpu().data.numpy(),loss1.cpu().data.numpy(),loss2.cpu().data.numpy(),
    loss3.cpu().data.numpy(),loss4.cpu().data.numpy(),loss5.cpu().data.numpy(),loss6.cpu().data.numpy()))
    return loss0, loss

#获取图片路径
def get_fileNames(rootdir):
    train_list = []
    label_list = []
    for root, dirs, files in os.walk(os.path.join(rootdir,"clip_img"),topdown = True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == ".jpg" or ending == ".png":
                pic_path = os.path.join(root,name)
                train_list.append(pic_path)
                label_list.append(pic_path.replace("clip_img", "matting").replace("clip","matting").replace(".jpg", ".png"))
    return np.array(train_list), np.array(label_list)


import torch
from torch.autograd import Function
import torch.nn.functional as F
from tqdm import tqdm
def eval_net(net, loader, device, n_val, writer, epoch):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['label']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            mask_pred = net(imgs)
            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])
        print(tot/n_val)
        if writer:
            writer.add_scalar('contact_ratio_val', tot/n_val, epoch)
    return tot/n_val


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter)/(self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).to(deivce).zero_()
    for i, c in enumerate(zip(input.detach().cpu(), target.detach().cpu())):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)



def train():
    # ------- 2. set the directory of training dataset --------

    batch_size_val = 1
    val_num = 0
    img_name_list,lbl_name_list = get_fileNames(rootdir)
    #划分训练集和测试集
    #划分训练集和测试集
    index = range(len(img_name_list))
    # np.random.shuffle(index)
    tv = int(len(index)*0.9)
    tr = int(tv*0.1)
    import random
    trtr = random.sample(index,tv)
    tftf = random.sample(trtr,tr)
    print(trtr)
    print(tftf)
    train_img_name_list, train_lbl_name_list = img_name_list[trtr], lbl_name_list[trtr]
    test_img_name_list, test_lbl_name_list = img_name_list[tftf], lbl_name_list[tftf]
    test_len = len(test_img_name_list)
    print("---",train_img_name_list)
    print("train images: ", len(train_img_name_list))
    print("train labels: ", len(train_lbl_name_list))
    print("---")
    train_num = len(train_img_name_list)

    salobj_dataset_train = SalObjDataset(
        img_name_list=train_img_name_list,
        lbl_name_list=train_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(800),
            RandomCrop(600),
            ToTensorLab(flag=0)]))
    salobj_dataloader_train = DataLoader(salobj_dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=1)

    salobj_dataset_test = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(800),
            RandomCrop(600),
            ToTensorLab(flag=0)]))
    salobj_dataloader_test = DataLoader(salobj_dataset_test, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    net = U2NETP(3,1)
    # if pretrain_path is not None:
    #     print("pretrain!!!!!!!!!!!!!!!!!!!!")
    
    
    
    
    net = mlu_quantize.adaptive_quantize(net, steps_per_epoch=len(salobj_dataloader_train), bitwidth=16)
    # net.load_state_dict(torch.load("/workspace/volume/guojun/Train/Semantic_segmentation/output/mlu_best.pth"))
    net.to(deivce)
        
    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = ct.to(optimizer,device=deivce)
    
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2 # save the model every 2000 iterations
    best_ratio = 0
    a =0
    for epoch in range(0, epoch_num):
        net.train()
        for i, data in enumerate(salobj_dataloader_train):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels = data['image'], data['label']
            # inputs = inputs.type(torch.FloatTensor)
            # labels = labels.type(torch.FloatTensor)
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
            else:
                inputs_v, labels_v = inputs.to(deivce),labels.to(deivce)
            # y zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = net(inputs_v)
            d0, d1, d2, d3, d4, d5, d6 = output
            loss0, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss.backward()
            optimizer.step()
            # # print statistics
            running_loss += loss
            running_tar_loss += loss0
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, target loss: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            # writer.add_scalar('train_l0', loss0, ite_num)
            # writer.add_scalar('train_avg', running_loss / ite_num4val, ite_num)
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss0, loss
            if ite_num % save_frq == 0:
                correct_ratio = eval_net(net, salobj_dataloader_test, deivce, test_len, None, ite_num)
                if correct_ratio>best_ratio:
                    best_ratio = correct_ratio
                    torch.save(net.state_dict(), model_dir)
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

            a+=1
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--data_path', type=str, default='./data_file', help='data file path(s)')
    parser.add_argument('--model_save_path',  type=str, default='./Inference_Mlu/model/', help='model.pth path(s)')
    parser.add_argument('--epoch_num',  type=int, default=2000, help='trainning  number of epoch')
    parser.add_argument('--batch_num', type=int, default=4, help='trainning data batch ')
    opt = parser.parse_args()
    # writer = SummaryWriter(logdir='./result')   # 训练过程数据存放在这个文件夹下
    #模型保存的目录路径
    model_dir = opt.model_save_path
    #图像数据集路径
    rootdir = opt.data_path
    epoch_num = opt.epoch_num
    batch_size_train = opt.batch_num
    train()
