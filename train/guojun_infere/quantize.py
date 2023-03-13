import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
from data_loader import RescaleT,RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NETP
import cv2
import time
import torch_mlu
import torch
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import torchvision.models as models
import argparse

#数据归一化
def normPredict(output):
    max_value = torch.max(output)
    min_value = torch.min(output)
    result = (output-min_value)/(max_value-min_value)
    return result
#数据后处理并保存

def save_output(image_name,pred,data_test,save_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    result =  np.insert(image ,3,pb_np[:,:,0],axis=-1)
    im = Image.fromarray(result)
    pb_np[np.where(pb_np>80)] = 255
    pb_np[np.where(pb_np<=80)] = 0
    masked = cv2.bitwise_and(image, image, mask=pb_np[:,:,0])
    cv2.imwrite(os.path.join(save_dir,image_name[image_name.rfind("/")+1:]),masked)


def quant_mode(model_dir, net, save_model_dir=None):
    #量化网络
    quantized_model = mlu_quantize.quantize_dynamic_mlu(net)
    #加载模型参数到cpu上
    param_state = torch.load(model_dir,map_location='cpu')
    #网络模型加载训练好的参数
    net.load_state_dict(param_state, strict=True)
    #模型量化
    quantized = mlu_quantize.quantize_dynamic_mlu(net, qconfig_spec={'iteration':1,'use_avg':False, 'data_scale':1.0, 
                                                                    'mean':[0,0,0], 'std':[1,1,1], 
                                                                    'firstconv':False, 'per_channel':False}, 
                                                                    dtype='int16', mapping=None, inplace=False, gen_quant=True)
    params_model = quantized.state_dict()
    #已量化的网络加载已量化的参数
    quantized_model.load_state_dict(params_model, strict=True)
    #保存量化后的模型参数
    if save_model_dir is not None:
        torch.save(params_model,  save_model_dir)
    return quantized_model

def load_quant_model(quant_model_dir, net):
        #加载量化后的模型参数
        param_state = torch.load(quant_model_dir)
        #量化网络
        quantized_model = mlu_quantize.quantize_dynamic_mlu(net)
        #加载参数到网络中
        quantized_model.load_state_dict(param_state, strict=True)
        return quantized_model

def main():
    # --------- 1. get image path and name ---------
    model_dir1 = os.path.join(os.getcwd(),'Inference_Mlu/model',"u2net_best.pth")
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--mode', nargs='+', type=int, default=1, help='data file path(s)')
    parser.add_argument('--image_dir', nargs='+', type=str, default='./test_data/test', help='data file path(s)')
    parser.add_argument('--source_model', nargs='+', type=str, default='./model/u2net_torch12.pth', help='model.pth path(s)')
    parser.add_argument('--save_result_dir', nargs='+', type=str, default='./test_data/u2netp_results', help='save data path')
    parser.add_argument('--save_quant_model_path', nargs='+', type=str, default='./model/quant_model.pth', help='trainning  number of epoch')
    opt = parser.parse_args()
    image_dir = opt.image_dir[0]
    img_name_list = glob.glob(image_dir + os.sep + '*')
    prediction_dir = opt.save_result_dir[0]   #输出结果保存
    model_dir =  opt.source_model[0]
    save_model_dir = opt.save_quant_model_path[0]
    mlu_mode_state = opt.mode[0]
    jit_save = False
    # --------- 2. dataloader ---------
    #  数据生成器
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list  ,  lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(800)   ,  ToTensorLab(flag=0)]))
    #  数据迭代器
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,
                                        shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    net = U2NETP(3,1)
    model_inference = None  
    #量化模型并逐层推理
    if mlu_mode_state == 1:
        model_inference = quant_mode(model_dir, net, save_model_dir=save_model_dir).to(ct.mlu_device())
        exit()
    #加载量化后的模型进行逐层推理
    elif mlu_mode_state == 2:
        #设置参与计算的内核数量
        core_num = 1
        if "MLU270" in os.popen("cnmon").read():
            ct.set_core_version("MLU270")
            core_num = min(core_num, 16)
        elif "MLU220" in os.popen("cnmon").read():
            ct.set_core_version("MLU220")
            core_num = min(core_num, 4)
        else:
            print("error: not set run device!!!")
        ct.set_core_number(core_num)
        model_inference = load_quant_model(save_model_dir, net).to(ct.mlu_device())
        
    #融合模式
    elif mlu_mode_state == 3:
        quantized_model = load_quant_model(save_model_dir, net)
        #设置固定权值梯度
        torch.set_grad_enabled(False)
        #设置参与计算的内核数量
        core_num = 1
        if "MLU270" in os.popen("cnmon").read():
            ct.set_core_version("MLU270")
            core_num = min(core_num, 16)
        elif "MLU220" in os.popen("cnmon").read():
            ct.set_core_version("MLU220")
            core_num = min(core_num, 4)
        else:
            print("error: not set run device!!!")
        ct.set_core_number(core_num)
        #设置输入数据形状和数据类型
        trace_input = torch.randn(1, 3, 800, 800,dtype=torch.float)
        #生成静态图
        model_inference = torch.jit.trace(quantized_model.to(ct.mlu_device()), trace_input.to(ct.mlu_device()), check_trace = False).to(ct.mlu_device())
    #cpu端运行
    else:
        model_inference = load_quant_model(save_model_dir, net)
    # model_inference.eval().float()
    # --------- 4. inference for each image ---------
    time_sum = 0.0
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image'].float()
        #如果是逐层或融合模式则将数据加载到mlu上
        if mlu_mode_state<4:
            inputs_test = inputs_test.to(ct.mlu_device())
        #模型推理
        start = time.time()
        d1,d2,d3,d4,d5,d6,d7 = model_inference(inputs_test)
        print(d1.shape)
        time_sum += (time.time()-start)
        # 获取最后一层的输出结果
        pred = d1[:,0,:,:]
        print(pred.shape)
        if mlu_mode_state<4:
            #将输出结果拷到cpu上
            pred = pred.cpu()
        #对输出结果进行标准化
        pred = normPredict(pred)
        # 输出数据后处理并保存到文件
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,data_test['image'],prediction_dir)
    print("run time:",time_sum)

if __name__ == "__main__":
    main()
