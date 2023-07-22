import logging
import os
import sys
from glob import glob
import cv2

import torch
from PIL import Image

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
)
from monai.networks.nets import UNet


from pathlib import Path
base = Path(os.environ['raw_data_base']) if 'raw_data_base' in os.environ.keys() else Path('./data')
assert base is not None, "Please assign the raw_data_base(which store the training data) in system path "
dir_test = base / 'test/test_2'
dir_checkpoint = 'checkpoints/'
Unet_Type = "AttentionUNet" # "UNet" or "AttentionUNet"

def inference_monai():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    images = sorted(glob(os.path.join(dir_test, "*.png"))) # all in list filenames

    train_imtrans = Compose( # 输入模型的图片的预处理
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            Resize((512, 512)), # 必须要加入这个，否则会报错，这里相当于直接拉伸，跟training保持一致
            ScaleIntensity(), # 其实就是归一化
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    data = train_imtrans(images)

    if Unet_Type == "UNet": 

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        model.load_state_dict(torch.load("best_metric_model_Unet.pth"))

    elif Unet_Type == "AttentionUNet":

        model = monai.networks.nets.AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),  
            kernel_size=3,
        ).to(device)   
        model.load_state_dict(torch.load("best_metric_model_AttentionUnet.pth"))
        

    model.eval()

    tf = Compose( # 恢复到原来的大小
    [   
        # Transpose((1, 2, 0)),
        Resize((657, 671)),
    ]
    )
    
    with torch.no_grad():
        for img in data:  # 其实相当于dataloader，因为也使用了Compose的组织方式

            img = img.to(device) # torch.Size([1, 512, 512])   HWC to CHW：img_trans = img_nd.transpose((2, 0, 1))
            img = img.unsqueeze(0) # torch.Size([1, 1, 512, 512]) unsqueeze扩增维度
            img = img.transpose(-1,-2) # 没问题了！
            output = model(img)
            result = post_trans(output) # torch.Size([1, 1, 512, 512])

            probs = result.squeeze(0) # squeeze压缩维度 torch.Size([1, 512, 512])
            probs = tf(probs.cpu()) # 重新拉伸到原来的大小
            full_mask = probs.squeeze().cpu().numpy() # return in cpu  # 
            # print(full_mask.shape) # (657, 671)
            
            
            cv2.imshow("result", full_mask*255)
            k = cv2.waitKey(5)
            if k==27:    # Esc key to stop
                break

            # pred = inferer(inputs=img, network=model) # 跟直接使用model()来推理效果一样
    

if __name__ == "__main__":
    inference_monai()