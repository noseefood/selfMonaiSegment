import sys
import rospy
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.dataset import BasicDataset
from unet import UNet
from torchvision import transforms

debug = True

class filter2DUS():
    def __init__(self):
        pass
    def rbox_length(self, mask):
        '''Using segmented mask to calculate the length of the needle'''

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # 很明显直接使用boundingRect上的边界点鲁棒性更好,minAreaRect似乎需要把中间连同
        if len(cnts) != 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0] # contourArea注意不是最大轮廓面积
            rbox = cv2.minAreaRect(cnt)  # for rbox (x, y), (width, height), angle = rect
            (center, (w,h), angle) = rbox # take it apart
            length = max(w, h)

            # self.line_ransac(mask, cnts)

            if debug == True:
                pts = cv2.boxPoints(rbox).astype(np.int32) # 4个点的图像坐标 pts:  [[411 175],[413 168],[539 202],[537 210]]  
                cv2.drawContours(mask, [pts], -1, (255, 255, 0), 1, cv2.LINE_AA) # 可以省略
                cv2.imshow('rbox on mask', mask) 
            return length, angle, rbox
        else:
            return 0, 0, None
        
    def line_ransac(self, mask, cnts):


        # # then apply fitline() function
        # [vx,vy,x,y] = cv2.fitLine(cnts[0],cv2.DIST_L2,0,0.01,0.001)

        # # Now find two extreme points on the line to draw line
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((mask.shape[1]-x)*vy/vx)+y)
        

        # #Finally draw the line
        # color = (0, 255, 255)
        # cv2.line(mask,(mask.shape[1]-1,righty),(0,lefty),color,5)

        # if debug == True:
        #     cv2.imshow('line on mask', mask)

        pass

    def bbox_length(self, mask):

        x,y,w,h = cv2.boundingRect(mask)
        mask = cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,0),2) # 2是线的宽度
        
        cv2.imshow('bbox on mask', mask)

        
    def filter(self, img):
        # 均值滤波
        img_mean = cv2.blur(img, (5,5))

        # 高斯滤波
        img_Guassian = cv2.GaussianBlur(img,(5,5),0)

        # 中值滤波
        img_median = cv2.medianBlur(img, 5)

        # 双边滤波
        img_bilater = cv2.bilateralFilter(img,9,75,75)

        # 展示不同的图片
        titles = ['srcImg','mean', 'Gaussian', 'median', 'bilateral']
        imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

        cv2.imshow('filter_mean', img_mean)
        cv2.imshow('filter_Guassian', img_Guassian)
        cv2.imshow('filter_median', img_median)
        cv2.imshow('filter_bilater', img_bilater)

        # 直方图均衡化equalizeHist
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        result1 = np.hstack((gray, equ))
        cv2.imshow('equ', result1)
        # CLAHE 自适应均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        result2 = np.hstack((gray, cl1))
        cv2.imshow('clahe', result2)

        # Laplacian算子
        imgLaplace2 = cv2.Laplacian(img, -1, ksize=3)
        imgRecovery = cv2.add(img, imgLaplace2)  # fusion
        cv2.imshow('Laplacian', imgRecovery)

    def filter_fusion(self, img):

        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # gray = cv2.threshold(gray, 60, 255, cv2.THRESH_TOZERO)[1]
        # cv2.imshow('threshold', gray)

        # img_mean = cv2.blur(gray, (10,10))
        # cv2.imshow('filter_mean', img_mean)

        # equ = cv2.equalizeHist(gray)
        # cv2.imshow('equ', equ) 
             
        # # 非锐化掩蔽
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # img_mean = cv2.blur(gray, (10,10))
        # cv2.imshow('filter_mean', img_mean)      

        # template =  gray - img_mean

        # result = gray + template

        # cv2.imshow('filter_fusion', result)
        pass

    def fre_filter(self, input_img):

        f_plot = self.fourier_plot(input_img)
        cv2.imshow('spec', f_plot)

        # lowpass filter
        img_rst = cv2.blur(input_img,(8,8))
        cv2.imshow('lowpass', img_rst)
        f_plot_rst = self.fourier_plot(img_rst)
        cv2.imshow('spec_rst', f_plot_rst)

        # highpass filter
        imgLaplace = cv2.Laplacian(input_img, -1, ksize=9)
        cv2.imshow('highpass', imgLaplace)
        f_plot_Laplace = self.fourier_plot(imgLaplace)
        cv2.imshow('spec_Laplace', f_plot_Laplace)

    def fourier_plot(self, input):

        img = input.copy()

        img = img.astype(np.float32) / 255

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to floats and do dft saving as complex output
        dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)

        # apply shift of origin from upper left corner to center of image 默认结果中心点位置是在左上角  逆过程是 ifftshift
        dft_shift = np.fft.fftshift(dft)

        # # extract magnitude and phase images(长度/角度)  cv2.cartToPolar() 类似cv2.magnitude
        # mag, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])
        # # get spectrum
        # spec = np.log(mag) / 20
        # cv2.imshow('spec', spec)

        result = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        spec = np.log(result) / 20
        # cv2.imshow('spec', spec)

        


        return spec


        






class NetworkInference(): 

    # 2D tracking
    model_path = '/home/xuesong/CAMP/US_servoing/src/py_client/scripts/unet/water.pth'

    def __init__(self, mode = "water"):
        self.net = UNet(n_channels=3, n_classes=1)

        if mode == "water":
            self.model = '/home/xuesong/CAMP/US_servoing/src/py_client/scripts/unet/water.pth'
            print("water checkpoint loaded")
        elif mode == "pork":
            self.model = '/home/xuesong/CAMP/US_servoing/src/py_client/scripts/unet/end_2.pth'
            print("pork checkpoint loaded")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device = self.device)
        self.net.load_state_dict(torch.load(self.model, map_location = self.device))
        self.net.eval() # inference mode

        self.scale_factor = 0.5
        self.threshold = 0.5

    def start(self, name):
        self.stop()
        pass

    def stop(self):
        pass

    def inference(self, full_img):
        # full_img: cv2.Mat(numpy.ndarray)

        preprocess_img = torch.from_numpy(BasicDataset.preprocess(full_img, self.scale_factor))  # 注意这里也跟训练一样进行了预处理(归一化)
        preprocess_img = preprocess_img.unsqueeze(0)
        preprocess_img = preprocess_img.to(device = self.device, dtype=torch.float32)

        with torch.no_grad(): # 不需要反向传播，不需要计算梯度

            output = self.net(preprocess_img)   # NN网络的直接相关只有这里

            if self.net.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img.shape[0]),
                    transforms.ToTensor()
                ]
            )
            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()
            mask = full_mask > self.threshold
            result = np.asarray((mask * 255).astype(np.uint8)) # 将mask(0,1)转换为0-255的灰度图用于可视化

        return result


class BasicUSPlayer():
    def __init__(self, mode = "water"):

        if mode == "water": # water
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/water_tank.avi')
            self.net = NetworkInference("water")
        elif mode == "pork-missaligen": # insertion-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_insertion_newProk19.avi')
            self.net = NetworkInference("pork")
        elif mode == "pork-3Dsegmentaion": # compounding-needle
            self.VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_compounding.avi')
            self.net = NetworkInference("pork")
        self.num_frames =  self.VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_counter = 0 
        self.ControlSpeedVar = 50
        self.HiSpeed = 100

        self.filter = filter2DUS()

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
            mask = self.net.inference(frame)
            length, angle, rbox = self.filter.rbox_length(mask)
            mask_copy = mask.copy()
            self.filter.bbox_length(mask_copy)
            
            # cv2.imshow('2D filter', self.net.inference(frame))
            # self.filter.filter(frame)
            self.filter.filter_fusion(frame)
            self.filter.fre_filter(frame)
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