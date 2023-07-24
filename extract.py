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

VideoCap = cv2.VideoCapture('/home/xuesong/CAMP/dataset/video_sim/pork_insertion_newProk19.avi')
num_frames =  VideoCap.get(cv2.CAP_PROP_FRAME_COUNT)

