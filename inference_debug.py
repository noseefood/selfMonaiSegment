import sys
import rospy
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
from monai.networks.nets import UNet, AttentionUnet

train_imtrans = Compose( # 输入模型的图片的预处理
    [   
        AddChannel(),  # 增加通道维度
        Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
        ScaleIntensity(), # 其实就是归一化
    ]
)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
tf = Compose( # 恢复到原来的大小
[   
    # Transpose((1, 2, 0)),
    Resize((657, 671)),
]
)


class NetworkInference():
    def __init__(self, mode = "water"):
        monai.config.print_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = AttentionUnet(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),  
        #     kernel_size=3,
        # ).to(self.device)   
        # self.model.load_state_dict(torch.load("best_metric_model_AttentionUnet.pth"))
        self.model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)
        self.model.load_state_dict(torch.load("best_metric_model_Unet.pth"))
        self.model.eval()

    def inference(self, img):
        with torch.no_grad():
            img = train_imtrans(img) # compose会自动返回tensor torch.Size([1, 512, 512])
            # print(img)

            img = img.to(self.device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            
            # img = img.transpose(-1,-2) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意这里与inference.py不同，这里不需要转置，以为读取图片的方式不一样！
            output = self.model(img)
            result = post_trans(output) # torch.Size([1, 1, 512, 512])

            probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            probs = tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            
            cv2.imshow("full_mask", full_mask)
            cv2.waitKey(0)

    pass


class BasicUSPlayer():
    def __init__(self, mode = "water"):

        if mode == "water": # water
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/water_tank.avi')
            self.net = NetworkInference("water")
        elif mode == "pork-missaligen": # insertion-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_insertion_newProk17.avi')
            self.net = NetworkInference("pork")
        elif mode == "pork-3Dsegmentaion": # compounding-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_compounding.avi')
            self.net = NetworkInference("pork")

        self.num_frames =  self.VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_counter = 0 
        self.ControlSpeedVar = 50
        self.HiSpeed = 100

    def start(self):

        print("All frames number:", self.num_frames)

        while(True):

            ret, frame = self.VideoCap.read()

            self.frame_counter += 1
            if self.frame_counter == int(self.VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)):
                self.frame_counter = 0
                self.VideoCap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            cv2.imshow('frame', frame)
            # hist = cv2.calcHist([frame], [0], None, [256], [0, 256])


            ###############################
            # 2D filter
            self.net.inference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            ###############################

            k = cv2.waitKey(30) & 0xff
            if k == 27: # ESC key to exit
                break
            cv2.waitKey(self.HiSpeed-self.ControlSpeedVar+1)
        cv2.destroyAllWindows()
        self.VideoCap.release()

if __name__ == "__main__":
    US_mode = "pork-missaligen" # "water" "pork-missaligen" "pork-3Dsegmentaion"
    usPlayter = BasicUSPlayer(mode = US_mode)
    usPlayter.start()