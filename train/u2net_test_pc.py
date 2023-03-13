import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import numpy as np
from PIL import Image
import glob
from Inference_Mlu.data_loader import RescaleT

from Inference_Mlu.data_loader import ToTensorLab
from Inference_Mlu.data_loader import SalObjDataset

from Inference_Mlu.model.u2net import U2NETP 
import argparse
import cv2 
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn
def save_output(image_name,pred,save_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    result =  np.insert(image ,3,pb_np[:,:,0],axis=-1)
    im = Image.fromarray(result)
    pb_np[np.where(pb_np>40)] = 255
    pb_np[np.where(pb_np<=40)] = 0
    #在模板mask上，将image和image做“与”操作
    masked = cv2.bitwise_and(image, image, mask=pb_np[:,:,0])
    cv2.imwrite(os.path.join(save_dir,image_name[image_name.rfind("/")+1:]),masked)
#content_img：背景图；
#type_img：人像图片；
#mask_img：网络模型推理的结果（掩模图）
def deal_pic(content_img, mask_img):
    #在模板mask上，将image和image做“与”操作得到图2.15的右图部分
    type_img= cv2.bitwise_and(image, image, mask=mask_img[:,:,0]) 
    #获取背景图尺寸大小
    size_content = content_img.shape
    #获取人像图尺寸大小
    size_type = type_img.shape
    #用于填充人想图的上边部分和右边部分，使人像图处于背景图左下角位置
    append_top = np.zeros((size_content[0]-size_type[0],size_content[1],3),type_img.dtype)
    append_right = np.zeros((size_type[0],size_content[1]-size_type[1],3),type_img.dtype)
    change1 = np.hstack((type_img,append_right))
    change2 = np.vstack((append_top,change1))
    #将人像图像素值大于0的像素赋值到背景图中
    content_img[np.where(change2>0)] = change2[np.where(change2>0)] 
    return content_img
def main():
    # --------- 1. get image path and name ---------
    image_dir = opt.data_path
    img_name_list = glob.glob(image_dir + os.sep + '*')
    prediction_dir = opt.save_data_path
    model_dir =  opt.model_dir
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
       		transform=transforms.Compose([RescaleT(320),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    # --------- 3. model define ---------
    net = U2NETP(3,1)

    net.load_state_dict(torch.load(model_dir,map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)
        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data_path', type=str, default='./Inference_Mlu/test_data/test', help='data file path(s)')
    parser.add_argument('--model_dir',  type=str, default='./Inference_Mlu/model/u2net.pth', help='model.pth path(s)')
    parser.add_argument('--save_data_path',type=str, default='./Inference_Mlu/test_data/u2netp_results', help='trainning  number of epoch')
    opt = parser.parse_args()
    main()
