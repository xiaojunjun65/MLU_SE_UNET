from __future__ import division
import torch
import torchvision.models as models
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import os
import sys
from model import U2NETP
import argparse

def load_quant_model(quant_model_dir, net):
        #加载量化后的模型参数
        param_state = torch.load(quant_model_dir)
        #量化网络
        quantized_model = mlu_quantize.quantize_dynamic_mlu(net)
        #加载参数到网络中
        quantized_model.load_state_dict(param_state, strict=True)
        return quantized_model
        
#生成离线模型
def genoff(model_params_path, save_offline_model_path, batch_size, core_number, in_heigth, in_width):
    #创建网络模型对象
    net = U2NETP(3, 1)
    #加载量化模型
    net = load_quant_model(model_params_path, net)
    #制作一个输入数据
    example_mlu = torch.randn(batch_size, 3, in_heigth, in_width,dtype=torch.float)
    #设置离线模型保存路径
    ct.save_as_cambricon(save_offline_model_path)
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
    #生成静态图
    net_traced = torch.jit.trace(net.to(ct.mlu_device()),
                                 example_mlu.to(ct.mlu_device()),
                                 check_trace=False)
    #融合模式推理，并将离线模型保存下来
    net_traced(example_mlu.to(ct.mlu_device()))
    ct.save_as_cambricon("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='genoff.py')
    parser.add_argument('--model_params_path', nargs='+', type=str, default='/model/quant_model.pth', help='quant model path(s)')
    parser.add_argument('--save_offline_model_path', nargs='+', type=str, default='/model/quant_model.cambricon', help='offline model path(s)')
    parser.add_argument('--core_number', nargs='+', type=int, default=1, help='core number')
    parser.add_argument('--in_heigth', nargs='+', type=int, default=800, help='input data height')
    parser.add_argument('--in_width', nargs='+', type=int, default=800, help='input data width')

    opt = parser.parse_args()
    model_params_path =  opt.model_params_path[0]
    save_offline_model_path =  opt.save_offline_model_path[0]
    batch_size = 1
    core_number = opt.core_number[0]
    in_heigth = opt.in_heigth[0]
    in_width = opt.in_width[0]
    genoff(model_params_path, save_offline_model_path, batch_size, core_number, in_heigth, in_width)

