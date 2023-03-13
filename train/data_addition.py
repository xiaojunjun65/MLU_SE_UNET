import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageEnhance
import argparse
import random
#第一组变换：图像的亮度、饱和度、对比度、锐度调节
def randomColor(image):
    # 设置随机因子
    random_factor = np.random.randint(0, 16) / 12.
    # 调整图像的饱和度
    color_image = ImageEnhance.Color(Image.fromarray(image)).enhance(random_factor)
    random_factor = np.random.randint(10, 16) / 12.
    # 调整图像的亮度
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
    random_factor = np.random.randint(10, 15) / 12.
    # 调整图像对比度
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 16) / 12.
    # 调整图像锐度 
    return np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))

#第二组变换：图像翻转
def flip(image, label_image):
    #水平镜像
    xImg = cv2.flip(image,1) 
    #掩模图水平镜像
    xImg_label = cv2.flip(label_image,1) 
    return xImg,xImg_label

#第三组变换：图像旋转
def rotate(image, mask):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    #中心坐标  
    (h_mask, w_mask) = mask.shape[:2]
    center_mask = (w_mask // 2, h_mask // 2)
    #旋转角度
    random_factor = np.random.randint(0, 45)    
    #获取图像旋转参数
    M = cv2.getRotationMatrix2D(center, random_factor, 1.0)  
    #获取掩模图旋转参数
    M_mask = cv2.getRotationMatrix2D(center_mask, random_factor, 1.0)
    #根据参数对图像进行仿射变换（旋转） 
    rotated = cv2.warpAffine(image, M, (w, h))  
    rotated_mask = cv2.warpAffine(mask, M_mask, (w, h))
    #返回经过旋转后的图片
    return rotated, rotated_mask

#获取图片路径
def addition_data(rootdir,save_file_name):
    train_list = []
    label_list = []
    save_path = [os.path.join(rootdir,"clip_img",save_file_name), os.path.join(rootdir,"matting",save_file_name)]
    for root, dirs, files in os.walk(os.path.join(rootdir,"clip_img"),topdown = True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == ".jpg" or ending == ".png":
                pic_path = os.path.join(root,name)
                img_data = cv2.imread(pic_path)
                img_label = cv2.imread(pic_path.replace("clip_img", "matting").replace("clip","matting").replace(".jpg", ".png"))
                if random.randint(0,1):
                    img_data,img_label = flip(img_data, img_label)
                img_data = randomColor(img_data)
                img_data,img_label = rotate(img_data,img_label)
                save_file(img_data,img_label,save_path,name)

def save_file(img_data,img_label,save_path,file_name):
    save_addition_data_path = save_path[0]     #原图
    save_addition_label_path = save_path[1]    #标签图
    if not os.path.exists(save_addition_data_path):
        os.makedirs(save_addition_data_path)
    if  not os.path.exists(save_addition_label_path):
        os.makedirs(save_addition_label_path)
    cv2.imwrite(os.path.join(save_addition_data_path,file_name),img_data)
    cv2.imwrite(os.path.join(save_addition_label_path,file_name),img_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--input_data_path',
                        help='picture files path',
                        default=None,
                        type=str)

    parser.add_argument('--data_save_folder_name',
                        help="",
                        default=None,
                        type=str)
    args = parser.parse_args()
    input_file_path = args.input_data_path
    save_file_name = args.data_save_folder_name
    addition_data(input_file_path,save_file_name)