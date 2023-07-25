import sys
import cv2
import os
import numpy as np
import open3d as o3d 
import copy

import pyransac3d as pyrsc
import SimpleITK as sitk


import torch
from PIL import Image

import logging
import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImage,
    ScaleIntensity,
    Resize,
    AddChannel,
)
from monai.transforms.transform import (  # noqa: F401
    apply_transform,
)
from monai.networks.nets import UNet, AttentionUnet


train_imtrans = Compose( # 输入模型的图片的预处理
    [
        # LoadImage(image_only=True, ensure_channel_first=True),  # 不需要会报错
        AddChannel(),  # 增加通道维度
        Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
        ScaleIntensity(), # 其实就是归一化
    ]
)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# tf = Compose( # 恢复到原来的大小
# [   
#     Resize((657, 671)),
# ]
# )

class VoxelSeg():

    def __init__(self, itkimage_unSeg):
        # self.voxelImg_unSeg = self.itk2voxel(itkimage_unSeg)
        self.itkimage_unSeg = itkimage_unSeg
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voxelImg_unSeg = self.itk2voxel(itkimage_unSeg)


    def itk2voxel(self, itkImg):
        # itk image to numpy array
        temp = sitk.GetArrayFromImage(itkImg)   # (154, 418, 449) z,y,x 转换为numpy (z,y,x): z:切片数量,y:切片宽,x:切片高
        return temp
    
    def model_load(self, Unet_Type):
        monai.config.print_config()
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if Unet_Type == "UNet": 

            self.model = UNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            ).to(self.device)
            self.model.load_state_dict(torch.load("best_metric_model_Unet.pth"))

        elif Unet_Type == "AttentionUNet":

            self.model = AttentionUnet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),  
                kernel_size=3,
            ).to(self.device)   
            self.model.load_state_dict(torch.load("best_metric_model_AttentionUnet.pth"))

        

    def process_and_replace_slices(self):
        # 获取图像的尺寸
        image = self.itkimage_unSeg
        output_file_path = "/home/xuesong/CAMP/segment/selfMonaiSegment/data/3D_seg/computed_segmented_Volume.mhd"
        size = image.GetSize()  # (449, 418, 154) x,y,z 注意直接读取itkimage(xyz)和转换为numpy(zyx)的区别
        # print(size)

        # # 沿z轴遍历每个切片，并进行修改
        self.model.eval()
        with torch.no_grad():
            for z in range(size[2]):
                # 提取单个切片
                slice_filter = sitk.ExtractImageFilter()
                slice_filter.SetSize([size[0], size[1], 0])
                slice_filter.SetIndex([0, 0, z])
                slice_image = slice_filter.Execute(image) # itk image

                # 在这里对切片进行修改，您可以添加您的图像处理代码
                img = sitk.GetArrayFromImage(slice_image)
                size_1, size_2 = img.shape
                tf = Compose( # 恢复到原来的大小
                [   
                    Resize((size_1, size_2)),
                    # ScaleIntensity(),

                ]
                )

                cv2.imshow("img", img)
                cv2.waitKey(0)    
                # print(img.shape) # (418, 429)

                img = train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])
                # print(img.shape) # torch.Size([1, 512, 512])

                img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
                img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度




                # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！




                output = self.model(img)
                result = post_trans(output) # torch.Size([1, 1, 512, 512]) # 二值化

                probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512]) 
                probs = tf(probs.cpu()) # 重新拉伸到原来的大小 注意一旦进行了resize，之前的二值化结果就会变成float32！！，在tf中已经进行了二值化，所以这里不需要了
                full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 

                full_mask = full_mask > 0
                full_mask = np.asarray((full_mask * 255).astype(np.uint8)) # 将mask(0,1)转换为0-255的灰度图用于可视化

                # full_mask = np.where(full_mask > 0.5, 1, 0) # 二值化
                # print(full_mask) # (418, 429)

                cv2.imshow("full_mask", full_mask)
                cv2.waitKey(0)
                # print(full_mask.shape) # (418, 429)
                
                result_itk = sitk.GetImageFromArray(full_mask)
                # print(result_itk.GetSize()) # (671, 657)
                self.voxelImg_unSeg[z,:,:] = full_mask 


        # 保存结果
        replaced_image = sitk.GetImageFromArray(self.voxelImg_unSeg)
        replaced_image.CopyInformation(self.itkimage_unSeg)
        sitk.WriteImage(replaced_image, output_file_path)
        


itkimage_unSeg = sitk.ReadImage("/home/xuesong/CAMP/segment/selfMonaiSegment/data/3D_seg/unsegmented_Volume.mhd")
extractor = VoxelSeg(itkimage_unSeg)
extractor.model_load("AttentionUNet")
extractor.process_and_replace_slices()