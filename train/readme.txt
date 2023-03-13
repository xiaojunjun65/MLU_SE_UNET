若需要进行数据增强和训练模型需注意以下几点：
1）图像数据需要是.jpg或.png格式，标签图像必须是.png格式；
2）训练图像数据可以放在./U2net_master/data_file/clip_img目录下，标签图像数据则放在./U2net_master/data_file/matting目录下；
3）进行数据增强后的图像数据会保存在所传的参数文件目录中，该目录会在./U2net_master/data_file/clip_img目录下生成。
(一）数据增强 
进入到./U2net_master下运行data_addition.py文件
$ python data_addition.py --input_data_path  ./data_file  --data_save_folder_name  addition_image
参数说明：
--input_data_path: 输入数据集目录路径，用于数据增强（clip_img和matting目录的父目录）
--data_save_folder_name: 数据增强后的结果存放的目录名称

（二）训练模型
进入到./U2net_master下运行u2net_train.py文件
python ttttt.py --data_path ./data_file --model_save_path /workspace/volume/guojun/Train/Semantic_segmentation/output/cpu_best.pth --epoch_num 20 --batch_num 4
$ python u2net_train.py --data_path ./data_file --model_save_path /workspace/volume/guojun/Train/Semantic_segmentation/output/mlu_best.pth --epoch_num 20 --batch_num 4
参数说明：
--data_path: 数据集目录路径
--model_save_path: 训练好的网络模型保存路径
--epoch_num: 要训练的周期数
--batch_num: 训练时使用的批次大小
训练时会将第0层的平均损失、各层之和的平均损失、验证集上的dice指标平均值记录在./U2net_master/result目录下的日志文件中，后面可用tensorboardx可视化
若模型在torch1.6及以上版本上训练，且模型后面需要在torch1.2上量化，则需在训练过程中torch.save()保存模型时设置参数_use_new_zipfile_serialization=False
训练过程中的信息都会保存在./U2net_master/result目录下，后续使用tensorboard进行可视化即可

（三）pc端进行推理
进入到./U2net_master下运行u2net_test_pc.py文件
$ python u2net_test_pc.py  --data_path ./Inference_Mlu/test_data/test --model_dir ./Inference_Mlu/model/u2net_torch12.pth --save_data_path ./Inference_Mlu/test_data/u2netp_results
参数说明：
--data_path  测试图像所在的目录路径
--model_dir  网络模型参数文件路径
--save_data_path  保存测试结果的目录路径

（四）在线推理
进入到./U2net_master/Inference_Mlu目录下运行u2net_test.py文件
量化模型
$ python u2net_test.py --mode 1 --source_model /workspace/volume/guojun/Train/Semantic_segmentation/output/mlu_best.pth  --save_quant_model_path   /workspace/volume/guojun/Train/Semantic_segmentation/output/qe_mlu_best.pth
在线逐层推理
$ python u2net_test.py --mode 2  --image_dir  /workspace/volume/guojun/Train/Semantic_segmentation/train/data_file/clip_img/  --save_result_dir  /workspace/volume/guojun/Train/Semantic_segmentation/output/   --save_quant_model_path  /workspace/volume/guojun/Train/Semantic_segmentation/output/qe_mlu_best.pth
在线融合推理
$ python u2net_test.py --mode 3  --image_dir  ./Inference_Mlu/offline_Inference/data --save_result_dir  ./test_data/u2netp_results --save_quant_model_path   ./model/quant_model.pth
cpu运行量化模型
$ python u2net_test.py --mode 4  --image_dir  ./Inference_Mlu/offline_Inference/data --save_result_dir  ./test_data/u2netp_results --save_quant_model_path   ./model/quant_model.pth
参数说明:
--mode 选择推理的模式，分别为1:量化模型；2:加载量化模型进行逐层推理；3:融合模式推理；4:量化模型并在CPU端进行推理
--image_dir  输入图像数据目录路径
--save_result_dir  保存输出结果目录路径
--source_model   原始模型路径
--save_quant_model_path 保存量化模型的路径

（五）生成离线模型
进入到./U2net_master/Inference_Mlu目录下运行genoff.py文件
$ python genoff.py --model_params_path ./model/quant_model.pth  --save_offline_model_path  ./offline_Inference/model  --core_number  1  --in_heigth  320  --in_width 320
参数说明：
--model_params_path  量化模型路径
--save_offline_model_path  保存生成的离线模型的路径
--core_number 设置离线推理所使用的核数量
--in_heigth  设置输入图像的高
--in_width 设置输入图像的宽

（六）离线推理
进入到./U2net_master/Inference_Mlu/offline_Inference/src目录下运行build.sh文件编译离线推理代码，可生成u2net_offline.out；
$ bash ./build.sh
然后运行可执行文件（u2net_offline.out）
$ ./u2net_offline.out -offmodel  ../model/quant_model.cambricon  -input_img_path ../data -save_path ../ -height 320  -width  320
参数说明：
-offmodel 离线模型的地址
-input_img_path 输入图像数据所在的目录路径
-save_path 保存图像输出结果所在的目录路径
-height  设置输入图像的高，和离线模型的高需一致
-width 设置输入图像的宽， 和离线模型的宽需一致

