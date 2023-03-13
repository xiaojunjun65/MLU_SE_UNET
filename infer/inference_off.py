import argparse
from python_off import ModelOffline
import cv2
from skimage import transform
from torchvision.transforms import transforms
import numpy as np
import torch_mlu.core.mlu_model as ct
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from data_loader import RescaleT,RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from PIL import Image
def normPredict(output):
    max_value = torch.max(output)
    min_value = torch.min(output)
    result = (output-min_value)/(max_value-min_value)
    return result
#数据后处理并保存

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
    pb_np[np.where(pb_np>80)] = 255
    pb_np[np.where(pb_np<=80)] = 0
    masked = cv2.bitwise_and(image, image, mask=pb_np[:,:,0])
    cv2.imwrite(os.path.join(save_dir,image_name[image_name.rfind("/")+1:]),masked)
import time
def main(opt):
    output_size =800
    img_name_list = glob.glob("/workspace/volume/guojun/Train/Semantic_segmentation/dataset/data_file/clip_img" + os.sep + '*')
    test_salobj_dataset = SalObjDataset(img_name_list =img_name_list   ,  lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(800)   ,  ToTensorLab(flag=0)]))
    #  数据迭代器
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,
                                        shuffle=False, num_workers=1)
    model = ModelOffline(opt.off_model,0)
    for i_test, data_test in enumerate(test_salobj_dataloader):
        start_time = time.clock()  # 程序开始时间
        img = data_test['image']
        img = np.transpose(img,(0,2,3,1))
        img = np.ascontiguousarray(img)
        img = np.array(img, dtype=np.float32) 
        d1,d2,d3,d4,d5,d6,d7 = model(img)
        end_time = time.clock()  # 程序结束时间
        run_time = end_time - start_time  # 程序的运行时间，单位为秒
        print("运行：", run_time)
        d1 = torch.from_numpy(np.ascontiguousarray(d1.transpose(0,3,1,2)))
        # 获取最后一层的输出结果
        pred = d1[:,0,:,:]
        pred = normPredict(pred)
        save_output(img_name_list[i_test],pred,"../")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='route_inference.py')
    parser.add_argument('--input_img', type=str, default='/workspace/volume/guojun/Train/Classification/dataset/test_cap/10.bmp', help='img not dir v1.0')
    parser.add_argument('--off_model', type=str, default='/workspace/volume/guojun/Train/Classification/offline/270cap_classification.cambricon', help='model_path')
    parser.add_argument("--input_size", type=int, default=100, help="model_size")
    opt = parser.parse_args()
    main(opt)
